use crate::RolloutRecorder;
use crate::compact::build_compacted_history;
use crate::compact::collect_user_messages;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::find_thread_path_by_id_str;
use crate::parse_turn_item;
use crate::protocol::CompactedItem;
use crate::protocol::EventMsg;
use crate::protocol::InitialHistory;
use crate::protocol::RolloutItem;
use crate::read_session_meta_line;
use codex_protocol::ThreadId;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseItem;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct MergeSource {
    pub thread_id: ThreadId,
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
struct EffectiveTurn {
    start_idx: usize,
    signature: Vec<ResponseItem>,
}

pub async fn build_merged_rollout(
    base_thread_id: ThreadId,
    base_path: &Path,
    merge_sources: &[MergeSource],
) -> CodexResult<Vec<RolloutItem>> {
    let codex_home = codex_home_from_rollout_path(base_path)?;
    let mut seen_source_ids = HashSet::new();
    for source in merge_sources {
        if !seen_source_ids.insert(source.thread_id) {
            return Err(CodexErr::InvalidRequest(format!(
                "duplicate merge source id {}",
                source.thread_id
            )));
        }
        if !is_descendant_of_thread(codex_home, source.path.as_path(), base_thread_id).await? {
            return Err(CodexErr::InvalidRequest(format!(
                "thread {} is not a descendant of {base_thread_id}",
                source.thread_id
            )));
        }
    }

    let mut merged_items = load_rollout_items(base_path).await?;
    merged_items.retain(|item| !matches!(item, RolloutItem::MergeBoundary(_)));
    let Some(RolloutItem::SessionMeta(meta_line)) = merged_items.first_mut() else {
        return Err(CodexErr::InvalidRequest(format!(
            "missing session metadata in base rollout `{}`",
            base_path.display()
        )));
    };
    meta_line.meta.merge_base_thread_id = None;
    meta_line.meta.merged_from_thread_ids = None;

    let ordered_sources = order_merge_sources(base_path, merge_sources).await?;
    for source in ordered_sources {
        let source_items = load_rollout_items(source.path.as_path()).await?;
        // Merge dedupe is intentionally based on the current logical merged history.
        // Once an earlier source compacts its prefix, later sources compare against that
        // lossy state rather than the original pre-compaction ancestry.
        let merged_turns = effective_turns(&merged_items);
        let source_turns = effective_turns(&source_items);
        let common_prefix_len = longest_common_prefix_len(&merged_turns, &source_turns);
        let Some(cut_idx) = source_suffix_start_idx(
            source_items.as_slice(),
            &merged_turns,
            &source_turns,
            common_prefix_len,
        ) else {
            continue;
        };
        merged_items.extend(
            source_items
                .into_iter()
                .skip(cut_idx)
                .filter(|item| !matches!(item, RolloutItem::MergeBoundary(_))),
        );
    }

    Ok(merged_items)
}

async fn load_rollout_items(path: &Path) -> CodexResult<Vec<RolloutItem>> {
    Ok(match RolloutRecorder::get_rollout_history(path).await? {
        InitialHistory::New => Vec::new(),
        InitialHistory::Resumed(resumed) => resumed.history,
        InitialHistory::Forked(items) => items,
    })
}

fn effective_turns(items: &[RolloutItem]) -> Vec<EffectiveTurn> {
    let mut turns = Vec::new();
    let mut active_turn: Option<EffectiveTurn> = None;
    let mut pending_start_idx: Option<usize> = None;

    for (idx, item) in items.iter().enumerate() {
        match item {
            RolloutItem::Compacted(compacted) => {
                if let Some(turn) = active_turn.take() {
                    turns.push(turn);
                }
                turns = effective_turns_after_compaction(&turns, compacted, idx);
                pending_start_idx = Some(idx);
            }
            RolloutItem::TurnContext(_) if active_turn.is_none() => {
                pending_start_idx.get_or_insert(idx);
            }
            RolloutItem::ResponseItem(response_item) => {
                if matches!(
                    parse_turn_item(response_item),
                    Some(TurnItem::UserMessage(_))
                ) {
                    if let Some(turn) = active_turn.take() {
                        turns.push(turn);
                    }
                    active_turn = Some(EffectiveTurn {
                        start_idx: pending_start_idx.take().unwrap_or(idx),
                        signature: vec![response_item.clone()],
                    });
                } else if let Some(turn) = active_turn.as_mut() {
                    turn.signature.push(response_item.clone());
                }
            }
            RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
                if let Some(turn) = active_turn.take() {
                    turns.push(turn);
                }
                let num_turns = usize::try_from(rollback.num_turns).unwrap_or(usize::MAX);
                turns.truncate(turns.len().saturating_sub(num_turns));
                pending_start_idx = None;
            }
            _ => {}
        }
    }

    if let Some(turn) = active_turn {
        turns.push(turn);
    }

    turns
}

fn effective_turns_after_compaction(
    turns: &[EffectiveTurn],
    compacted: &CompactedItem,
    start_idx: usize,
) -> Vec<EffectiveTurn> {
    let replacement_history = compacted.replacement_history.clone().unwrap_or_else(|| {
        let history: Vec<ResponseItem> = turns
            .iter()
            .flat_map(|turn| turn.signature.iter().cloned())
            .collect();
        let user_messages = collect_user_messages(&history);
        build_compacted_history(Vec::new(), &user_messages, &compacted.message)
    });

    effective_turns_from_response_items(&replacement_history, start_idx)
}

fn effective_turns_from_response_items(
    items: &[ResponseItem],
    start_idx: usize,
) -> Vec<EffectiveTurn> {
    let mut turns = Vec::new();
    let mut active_turn: Option<EffectiveTurn> = None;

    for item in items {
        if matches!(parse_turn_item(item), Some(TurnItem::UserMessage(_))) {
            if let Some(turn) = active_turn.take() {
                turns.push(turn);
            }
            active_turn = Some(EffectiveTurn {
                start_idx,
                signature: vec![item.clone()],
            });
        } else if let Some(turn) = active_turn.as_mut() {
            turn.signature.push(item.clone());
        }
    }

    if let Some(turn) = active_turn {
        turns.push(turn);
    }

    turns
}

fn longest_common_prefix_len(lhs: &[EffectiveTurn], rhs: &[EffectiveTurn]) -> usize {
    lhs.iter()
        .zip(rhs.iter())
        .take_while(|(lhs, rhs)| lhs.signature == rhs.signature)
        .count()
}

fn source_suffix_start_idx(
    items: &[RolloutItem],
    merged_turns: &[EffectiveTurn],
    source_turns: &[EffectiveTurn],
    common_prefix_len: usize,
) -> Option<usize> {
    let rollback_idx =
        first_rollback_that_removes_shared_turns(items, merged_turns, common_prefix_len);
    let turn_idx = (common_prefix_len < source_turns.len())
        .then(|| suffix_start_idx(items, source_turns[common_prefix_len].start_idx));

    match (rollback_idx, turn_idx) {
        (Some(rollback_idx), Some(turn_idx)) => Some(rollback_idx.min(turn_idx)),
        (Some(rollback_idx), None) => Some(rollback_idx),
        (None, Some(turn_idx)) => Some(turn_idx),
        (None, None) => None,
    }
}

fn first_rollback_that_removes_shared_turns(
    items: &[RolloutItem],
    merged_turns: &[EffectiveTurn],
    final_common_prefix_len: usize,
) -> Option<usize> {
    let mut turns = Vec::new();
    let mut active_turn: Option<EffectiveTurn> = None;
    let mut pending_start_idx: Option<usize> = None;
    let mut max_prefix_len_seen = 0;

    for (idx, item) in items.iter().enumerate() {
        match item {
            RolloutItem::Compacted(compacted) => {
                if let Some(turn) = active_turn.take() {
                    turns.push(turn);
                    max_prefix_len_seen =
                        max_prefix_len_seen.max(longest_common_prefix_len(merged_turns, &turns));
                }
                turns = effective_turns_after_compaction(&turns, compacted, idx);
                max_prefix_len_seen =
                    max_prefix_len_seen.max(longest_common_prefix_len(merged_turns, &turns));
                pending_start_idx = Some(idx);
            }
            RolloutItem::TurnContext(_) if active_turn.is_none() => {
                pending_start_idx.get_or_insert(idx);
            }
            RolloutItem::ResponseItem(response_item) => {
                if matches!(
                    parse_turn_item(response_item),
                    Some(TurnItem::UserMessage(_))
                ) {
                    if let Some(turn) = active_turn.take() {
                        turns.push(turn);
                        max_prefix_len_seen = max_prefix_len_seen
                            .max(longest_common_prefix_len(merged_turns, &turns));
                    }
                    active_turn = Some(EffectiveTurn {
                        start_idx: pending_start_idx.take().unwrap_or(idx),
                        signature: vec![response_item.clone()],
                    });
                } else if let Some(turn) = active_turn.as_mut() {
                    turn.signature.push(response_item.clone());
                }
            }
            RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) => {
                if let Some(turn) = active_turn.take() {
                    turns.push(turn);
                    max_prefix_len_seen =
                        max_prefix_len_seen.max(longest_common_prefix_len(merged_turns, &turns));
                }
                let num_turns = usize::try_from(rollback.num_turns).unwrap_or(usize::MAX);
                turns.truncate(turns.len().saturating_sub(num_turns));
                if max_prefix_len_seen > final_common_prefix_len
                    && longest_common_prefix_len(merged_turns, &turns) < max_prefix_len_seen
                {
                    return Some(idx);
                }
                pending_start_idx = None;
            }
            _ => {}
        }
    }

    None
}

fn suffix_start_idx(items: &[RolloutItem], start_idx: usize) -> usize {
    let mut idx = start_idx;
    while idx > 0
        && matches!(
            items.get(idx - 1),
            Some(RolloutItem::Compacted(_) | RolloutItem::TurnContext(_))
        )
    {
        idx -= 1;
    }
    idx
}

async fn order_merge_sources(
    base_path: &Path,
    merge_sources: &[MergeSource],
) -> CodexResult<Vec<MergeSource>> {
    let codex_home = codex_home_from_rollout_path(base_path)?;
    let mut ordered = Vec::new();

    for source in merge_sources.iter().cloned() {
        let mut insert_at = ordered.len();
        for (idx, existing) in ordered.iter().enumerate() {
            if is_ancestor_source(codex_home, &source, existing).await? {
                insert_at = idx;
                break;
            }
            if is_ancestor_source(codex_home, existing, &source).await? {
                insert_at = idx + 1;
            }
        }
        ordered.insert(insert_at, source);
    }

    Ok(ordered)
}

fn codex_home_from_rollout_path(rollout_path: &Path) -> CodexResult<&Path> {
    rollout_path.ancestors().nth(5).ok_or_else(|| {
        CodexErr::InvalidRequest(format!(
            "failed to derive codex home from rollout `{}`",
            rollout_path.display()
        ))
    })
}

async fn is_ancestor_source(
    codex_home: &Path,
    ancestor: &MergeSource,
    descendant: &MergeSource,
) -> CodexResult<bool> {
    is_descendant_of_thread(codex_home, descendant.path.as_path(), ancestor.thread_id).await
}

async fn is_descendant_of_thread(
    codex_home: &Path,
    descendant_path: &Path,
    ancestor_thread_id: ThreadId,
) -> CodexResult<bool> {
    let mut current_path = descendant_path.to_path_buf();
    let mut seen = HashSet::new();

    loop {
        let meta_line = read_session_meta_line(current_path.as_path()).await?;
        let Some(parent_thread_id) = meta_line.meta.forked_from_id else {
            return Ok(false);
        };
        if parent_thread_id == ancestor_thread_id {
            return Ok(true);
        }
        if !seen.insert(parent_thread_id) {
            return Ok(false);
        }
        let Some(parent_path) =
            find_thread_path_by_id_str(codex_home, &parent_thread_id.to_string()).await?
        else {
            return Ok(false);
        };
        current_path = parent_path;
    }
}

#[cfg(test)]
mod tests {
    use super::MergeSource;
    use super::build_merged_rollout;
    use super::effective_turns;
    use codex_protocol::ThreadId;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::protocol::CompactedItem;
    use codex_protocol::protocol::EventMsg;
    use codex_protocol::protocol::RolloutItem;
    use codex_protocol::protocol::RolloutLine;
    use codex_protocol::protocol::SessionMeta;
    use codex_protocol::protocol::SessionMetaLine;
    use codex_protocol::protocol::SessionSource;
    use codex_protocol::protocol::ThreadRolledBackEvent;
    use pretty_assertions::assert_eq;
    use std::path::Path;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[derive(Clone, Copy)]
    struct TestTurn<'a> {
        user: &'a str,
        assistant: &'a str,
    }

    fn thread_id() -> ThreadId {
        ThreadId::new()
    }

    fn write_rollout(
        root: &Path,
        thread_id: ThreadId,
        forked_from_id: Option<ThreadId>,
        turns: &[TestTurn<'_>],
    ) -> PathBuf {
        write_rollout_items(
            root,
            thread_id,
            forked_from_id,
            rollout_items_for_turns(turns),
        )
    }

    fn write_rollout_items(
        root: &Path,
        thread_id: ThreadId,
        forked_from_id: Option<ThreadId>,
        items: Vec<RolloutItem>,
    ) -> PathBuf {
        let dir = root.join("sessions").join("2026").join("01").join("01");
        std::fs::create_dir_all(&dir).expect("create rollout dir");
        let path = dir.join(format!("rollout-2026-01-01T00-00-00-{thread_id}.jsonl"));
        let mut lines = Vec::new();
        lines.push(RolloutLine {
            timestamp: "2026-01-01T00:00:00.000Z".to_string(),
            item: RolloutItem::SessionMeta(SessionMetaLine {
                meta: SessionMeta {
                    id: thread_id,
                    forked_from_id,
                    merge_base_thread_id: None,
                    merged_from_thread_ids: None,
                    timestamp: "2026-01-01T00:00:00.000Z".to_string(),
                    cwd: root.to_path_buf(),
                    originator: "test".to_string(),
                    cli_version: "0.0.0".to_string(),
                    source: SessionSource::Cli,
                    agent_nickname: None,
                    agent_role: None,
                    model_provider: Some("test".to_string()),
                    base_instructions: None,
                    dynamic_tools: None,
                    memory_mode: None,
                },
                git: None,
            }),
        });
        lines.extend(items.into_iter().map(|item| RolloutLine {
            timestamp: "2026-01-01T00:00:00.000Z".to_string(),
            item,
        }));
        std::fs::write(
            &path,
            lines
                .into_iter()
                .map(|line| serde_json::to_string(&line).expect("serialize"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("write rollout");
        path
    }

    fn rollout_items_for_turns(turns: &[TestTurn<'_>]) -> Vec<RolloutItem> {
        turns
            .iter()
            .flat_map(|turn| {
                [
                    RolloutItem::ResponseItem(response_message("user", turn.user)),
                    RolloutItem::ResponseItem(response_message("assistant", turn.assistant)),
                ]
            })
            .collect()
    }

    fn response_message(role: &str, text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: vec![match role {
                "assistant" => ContentItem::OutputText {
                    text: text.to_string(),
                },
                _ => ContentItem::InputText {
                    text: text.to_string(),
                },
            }],
            end_turn: None,
            phase: None,
        }
    }

    fn logical_turn_pairs(items: &[RolloutItem]) -> Vec<(String, Option<String>)> {
        effective_turns(items)
            .into_iter()
            .map(|turn| {
                let mut user = None;
                let mut assistant = None;
                for item in &turn.signature {
                    if let ResponseItem::Message { role, content, .. } = item {
                        let text = content
                            .iter()
                            .find_map(|content_item| match content_item {
                                ContentItem::InputText { text }
                                | ContentItem::OutputText { text } => Some(text.clone()),
                                ContentItem::InputImage { .. } => None,
                            })
                            .unwrap_or_default();
                        if role == "user" {
                            user = Some(text);
                        } else if role == "assistant" {
                            assistant = Some(text);
                        }
                    }
                }
                (user.unwrap_or_default(), assistant)
            })
            .collect()
    }

    #[tokio::test]
    async fn merge_rejects_non_descendant() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "base",
                assistant: "base answer",
            }],
        );
        let other_id = thread_id();
        let other_path = write_rollout(
            temp.path(),
            other_id,
            None,
            &[TestTurn {
                user: "other",
                assistant: "other answer",
            }],
        );

        let err = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: other_id,
                path: other_path,
            }],
        )
        .await
        .expect_err("merge should fail");

        assert_eq!(
            err.to_string(),
            format!("thread {other_id} is not a descendant of {base_id}")
        );
    }

    #[tokio::test]
    async fn merge_direct_child_appends_only_child_suffix() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "b2",
                    assistant: "a2",
                },
            ],
        );
        let child_id = thread_id();
        let child_path = write_rollout(
            temp.path(),
            child_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "b2",
                    assistant: "a2",
                },
                TestTurn {
                    user: "c1",
                    assistant: "c1 answer",
                },
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: child_id,
                path: child_path,
            }],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("b2".to_string(), Some("a2".to_string())),
                ("c1".to_string(), Some("c1 answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_grandchild_appends_full_descendant_suffix() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "b1",
                assistant: "a1",
            }],
        );
        let child_id = thread_id();
        write_rollout(
            temp.path(),
            child_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "c1",
                    assistant: "child answer",
                },
            ],
        );
        let grandchild_id = thread_id();
        let grandchild_path = write_rollout(
            temp.path(),
            grandchild_id,
            Some(child_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "c1",
                    assistant: "child answer",
                },
                TestTurn {
                    user: "g1",
                    assistant: "grandchild answer",
                },
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: grandchild_id,
                path: grandchild_path,
            }],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("g1".to_string(), Some("grandchild answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_child_and_grandchild_do_not_duplicate_shared_child_turns() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "b1",
                assistant: "a1",
            }],
        );
        let child_id = thread_id();
        let child_path = write_rollout(
            temp.path(),
            child_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "c1",
                    assistant: "child answer",
                },
            ],
        );
        let grandchild_id = thread_id();
        let grandchild_path = write_rollout(
            temp.path(),
            grandchild_id,
            Some(child_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "c1",
                    assistant: "child answer",
                },
                TestTurn {
                    user: "g1",
                    assistant: "grandchild answer",
                },
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: child_id,
                    path: child_path,
                },
                MergeSource {
                    thread_id: grandchild_id,
                    path: grandchild_path,
                },
            ],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("g1".to_string(), Some("grandchild answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_forces_ancestor_before_descendant_even_if_selected_reversed() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "b1",
                assistant: "a1",
            }],
        );
        let child_id = thread_id();
        let child_path = write_rollout(
            temp.path(),
            child_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "c1",
                    assistant: "child answer",
                },
            ],
        );
        let grandchild_id = thread_id();
        let grandchild_path = write_rollout(
            temp.path(),
            grandchild_id,
            Some(child_id),
            &[
                TestTurn {
                    user: "b1",
                    assistant: "a1",
                },
                TestTurn {
                    user: "c1",
                    assistant: "child answer",
                },
                TestTurn {
                    user: "g1",
                    assistant: "grandchild answer",
                },
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: grandchild_id,
                    path: grandchild_path,
                },
                MergeSource {
                    thread_id: child_id,
                    path: child_path,
                },
            ],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("g1".to_string(), Some("grandchild answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_siblings_preserve_user_order_without_interleaving() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "base",
                assistant: "base answer",
            }],
        );
        let first_id = thread_id();
        let first_path = write_rollout(
            temp.path(),
            first_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "base",
                    assistant: "base answer",
                },
                TestTurn {
                    user: "first",
                    assistant: "first answer",
                },
            ],
        );
        let second_id = thread_id();
        let second_path = write_rollout(
            temp.path(),
            second_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "base",
                    assistant: "base answer",
                },
                TestTurn {
                    user: "second",
                    assistant: "second answer",
                },
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: first_id,
                    path: first_path,
                },
                MergeSource {
                    thread_id: second_id,
                    path: second_path,
                },
            ],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base".to_string(), Some("base answer".to_string())),
                ("first".to_string(), Some("first answer".to_string())),
                ("second".to_string(), Some("second answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_uses_logical_state_after_compaction() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let shared_turn = TestTurn {
            user: "base",
            assistant: "base answer",
        };
        let base_path = write_rollout(temp.path(), base_id, None, &[shared_turn]);
        let branch_id = thread_id();
        let branch_path = write_rollout_items(temp.path(), branch_id, Some(base_id), {
            let mut items = rollout_items_for_turns(&[shared_turn]);
            items.push(RolloutItem::Compacted(CompactedItem {
                message: "summary".to_string(),
                replacement_history: Some(vec![response_message("user", "summary")]),
            }));
            items.extend(rollout_items_for_turns(&[TestTurn {
                user: "branch",
                assistant: "branch answer",
            }]));
            items
        });

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("summary".to_string(), None),
                ("branch".to_string(), Some("branch answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_compacted_source_before_sibling_replays_shared_prefix() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let shared_turn = TestTurn {
            user: "base",
            assistant: "base answer",
        };
        let base_path = write_rollout(temp.path(), base_id, None, &[shared_turn]);
        let compacted_id = thread_id();
        let compacted_path = write_rollout_items(temp.path(), compacted_id, Some(base_id), {
            let mut items = rollout_items_for_turns(&[shared_turn]);
            items.push(RolloutItem::Compacted(CompactedItem {
                message: "summary".to_string(),
                replacement_history: Some(vec![response_message("user", "summary")]),
            }));
            items.extend(rollout_items_for_turns(&[TestTurn {
                user: "first",
                assistant: "first answer",
            }]));
            items
        });
        let sibling_id = thread_id();
        let sibling_path = write_rollout(
            temp.path(),
            sibling_id,
            Some(base_id),
            &[
                shared_turn,
                TestTurn {
                    user: "second",
                    assistant: "second answer",
                },
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: compacted_id,
                    path: compacted_path,
                },
                MergeSource {
                    thread_id: sibling_id,
                    path: sibling_path,
                },
            ],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("summary".to_string(), None),
                ("first".to_string(), Some("first answer".to_string())),
                ("base".to_string(), Some("base answer".to_string())),
                ("second".to_string(), Some("second answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_preserves_rollback_of_inherited_turn_without_new_turns() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[
                TestTurn {
                    user: "base-1",
                    assistant: "base-1 answer",
                },
                TestTurn {
                    user: "base-2",
                    assistant: "base-2 answer",
                },
            ],
        );
        let branch_id = thread_id();
        let branch_path = write_rollout_items(temp.path(), branch_id, Some(base_id), {
            let mut items = rollout_items_for_turns(&[
                TestTurn {
                    user: "base-1",
                    assistant: "base-1 answer",
                },
                TestTurn {
                    user: "base-2",
                    assistant: "base-2 answer",
                },
            ]);
            items.push(RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
                ThreadRolledBackEvent { num_turns: 1 },
            )));
            items
        });

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![("base-1".to_string(), Some("base-1 answer".to_string()))]
        );
    }

    #[tokio::test]
    async fn merge_preserves_rollback_of_inherited_turn_before_new_turn() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[
                TestTurn {
                    user: "base-1",
                    assistant: "base-1 answer",
                },
                TestTurn {
                    user: "base-2",
                    assistant: "base-2 answer",
                },
            ],
        );
        let branch_id = thread_id();
        let branch_path = write_rollout_items(temp.path(), branch_id, Some(base_id), {
            let mut items = rollout_items_for_turns(&[
                TestTurn {
                    user: "base-1",
                    assistant: "base-1 answer",
                },
                TestTurn {
                    user: "base-2",
                    assistant: "base-2 answer",
                },
            ]);
            items.push(RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
                ThreadRolledBackEvent { num_turns: 1 },
            )));
            items.extend(rollout_items_for_turns(&[TestTurn {
                user: "branch",
                assistant: "branch answer",
            }]));
            items
        });

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base-1".to_string(), Some("base-1 answer".to_string())),
                ("branch".to_string(), Some("branch answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn merge_uses_logical_state_after_rollback() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "base",
                assistant: "base answer",
            }],
        );
        let branch_id = thread_id();
        let branch_path = write_rollout_items(temp.path(), branch_id, Some(base_id), {
            let mut items = rollout_items_for_turns(&[
                TestTurn {
                    user: "base",
                    assistant: "base answer",
                },
                TestTurn {
                    user: "branch",
                    assistant: "branch answer",
                },
            ]);
            items.push(RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
                ThreadRolledBackEvent { num_turns: 1 },
            )));
            items
        });

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[MergeSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("merge");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![("base".to_string(), Some("base answer".to_string()))]
        );
    }

    #[tokio::test]
    async fn merge_duplicate_source_ids_in_single_request_errors() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "base",
                assistant: "base answer",
            }],
        );
        let child_id = thread_id();
        let child_path = write_rollout(
            temp.path(),
            child_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "base",
                    assistant: "base answer",
                },
                TestTurn {
                    user: "child",
                    assistant: "child answer",
                },
            ],
        );

        let err = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: child_id,
                    path: child_path.clone(),
                },
                MergeSource {
                    thread_id: child_id,
                    path: child_path,
                },
            ],
        )
        .await
        .expect_err("merge should fail");

        assert_eq!(
            err.to_string(),
            format!("duplicate merge source id {child_id}")
        );
    }
}
