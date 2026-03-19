use crate::RolloutRecorder;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::protocol::InitialHistory;
use crate::protocol::MergeBoundaryItem;
use crate::protocol::RolloutItem;
use codex_protocol::ThreadId;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;

#[cfg(test)]
use crate::compact::build_compacted_history;
#[cfg(test)]
use crate::compact::collect_user_messages;
#[cfg(test)]
use crate::parse_turn_item;
#[cfg(test)]
use crate::protocol::CompactedItem;
#[cfg(test)]
use crate::protocol::EventMsg;
#[cfg(test)]
use codex_protocol::items::TurnItem;
#[cfg(test)]
use codex_protocol::models::ResponseItem;

#[derive(Debug, Clone)]
pub struct CombineSource {
    pub thread_id: ThreadId,
    pub path: PathBuf,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct EffectiveTurn {
    start_idx: usize,
    signature: Vec<ResponseItem>,
}

pub async fn build_combined_rollout(
    base_thread_id: ThreadId,
    base_path: &Path,
    combine_sources: &[CombineSource],
) -> CodexResult<Vec<RolloutItem>> {
    let mut seen_source_ids = HashSet::new();
    for source in combine_sources {
        if !seen_source_ids.insert(source.thread_id) {
            return Err(CodexErr::InvalidRequest(format!(
                "duplicate combine source id {}",
                source.thread_id
            )));
        }
    }

    let mut prepared_sources = Vec::with_capacity(combine_sources.len());
    for source in combine_sources {
        let source_items = load_rollout_items(source.path.as_path()).await?;
        let Some(RolloutItem::SessionMeta(meta_line)) = source_items.first() else {
            return Err(CodexErr::InvalidRequest(format!(
                "missing session metadata in combine source rollout `{}`",
                source.path.display()
            )));
        };
        if meta_line.meta.id != source.thread_id {
            return Err(CodexErr::InvalidRequest(format!(
                "combine source id {} does not match rollout session id {} for `{}`",
                source.thread_id,
                meta_line.meta.id,
                source.path.display()
            )));
        }
        prepared_sources.push((
            source.thread_id,
            InitialHistory::Forked(source_items.clone()).merged_from_thread_ids(),
            source_items.into_iter().skip(1).collect::<Vec<_>>(),
        ));
    }

    let mut merged_items = load_rollout_items(base_path).await?;
    let mut merged_from_thread_ids =
        InitialHistory::Forked(merged_items.clone()).merged_from_thread_ids();
    let mut seen_merged_from_thread_ids: HashSet<ThreadId> =
        merged_from_thread_ids.iter().copied().collect();
    for (source_thread_id, source_merged_from_thread_ids, _) in &prepared_sources {
        if seen_merged_from_thread_ids.insert(*source_thread_id) {
            merged_from_thread_ids.push(*source_thread_id);
        }
        for nested_thread_id in source_merged_from_thread_ids {
            if seen_merged_from_thread_ids.insert(*nested_thread_id) {
                merged_from_thread_ids.push(*nested_thread_id);
            }
        }
    }

    let Some(RolloutItem::SessionMeta(meta_line)) = merged_items.first_mut() else {
        return Err(CodexErr::InvalidRequest(format!(
            "missing session metadata in base rollout `{}`",
            base_path.display()
        )));
    };
    meta_line.meta.merge_base_thread_id =
        (!merged_from_thread_ids.is_empty()).then_some(base_thread_id);
    meta_line.meta.merged_from_thread_ids =
        (!merged_from_thread_ids.is_empty()).then_some(merged_from_thread_ids);
    for (source_thread_id, _, source_segment_items) in prepared_sources {
        merged_items.push(RolloutItem::MergeBoundary(MergeBoundaryItem {
            source_thread_id,
        }));
        merged_items.extend(source_segment_items);
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

#[cfg(test)]
fn effective_turns(items: &[RolloutItem]) -> Vec<EffectiveTurn> {
    let mut turns = Vec::new();
    let mut segment_items = Vec::new();

    for item in items {
        match item {
            RolloutItem::SessionMeta(_) | RolloutItem::MergeBoundary(_) => {
                if !segment_items.is_empty() {
                    turns.extend(effective_turns_in_segment(&segment_items));
                    segment_items.clear();
                }
            }
            _ => segment_items.push(item.clone()),
        }
    }

    if !segment_items.is_empty() {
        turns.extend(effective_turns_in_segment(&segment_items));
    }

    turns
}

#[cfg(test)]
fn effective_turns_in_segment(items: &[RolloutItem]) -> Vec<EffectiveTurn> {
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
mod tests {
    use super::CombineSource;
    use super::build_combined_rollout;
    use super::effective_turns;
    use codex_protocol::ThreadId;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::protocol::CompactedItem;
    use codex_protocol::protocol::EventMsg;
    use codex_protocol::protocol::MergeBoundaryItem;
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
    async fn combine_accepts_unrelated_thread() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: other_id,
                path: other_path,
            }],
        )
        .await
        .expect("combine should succeed");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base".to_string(), Some("base answer".to_string())),
                ("other".to_string(), Some("other answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_appends_direct_child_full_history() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: child_id,
                path: child_path,
            }],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("b2".to_string(), Some("a2".to_string())),
                ("b1".to_string(), Some("a1".to_string())),
                ("b2".to_string(), Some("a2".to_string())),
                ("c1".to_string(), Some("c1 answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_records_provenance_in_session_meta() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: other_id,
                path: other_path,
            }],
        )
        .await
        .expect("combine should succeed");

        let session_meta = merged
            .iter()
            .find_map(|item| match item {
                RolloutItem::SessionMeta(meta_line) => Some(&meta_line.meta),
                _ => None,
            })
            .expect("session meta");

        assert_eq!(session_meta.merge_base_thread_id, Some(base_id));
        assert_eq!(session_meta.merged_from_thread_ids, Some(vec![other_id]));
    }

    #[tokio::test]
    async fn combine_inserts_merge_boundary_before_each_source() {
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
            None,
            &[TestTurn {
                user: "first",
                assistant: "first answer",
            }],
        );
        let second_id = thread_id();
        let second_path = write_rollout(
            temp.path(),
            second_id,
            None,
            &[TestTurn {
                user: "second",
                assistant: "second answer",
            }],
        );

        let combined = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[
                CombineSource {
                    thread_id: first_id,
                    path: first_path,
                },
                CombineSource {
                    thread_id: second_id,
                    path: second_path,
                },
            ],
        )
        .await
        .expect("combine should succeed");

        let boundary_ids = combined
            .iter()
            .filter_map(|item| match item {
                RolloutItem::MergeBoundary(boundary) => Some(boundary.source_thread_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(boundary_ids, vec![first_id, second_id]);
    }

    #[tokio::test]
    async fn combine_records_nested_source_provenance_in_session_meta() {
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
        let nested_id = thread_id();
        let source_id = thread_id();
        let source_path = write_rollout_items(temp.path(), source_id, None, {
            let mut items = vec![RolloutItem::MergeBoundary(MergeBoundaryItem {
                source_thread_id: nested_id,
            })];
            items.extend(rollout_items_for_turns(&[TestTurn {
                user: "source",
                assistant: "source answer",
            }]));
            items
        });

        let combined = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: source_id,
                path: source_path,
            }],
        )
        .await
        .expect("combine should succeed");

        let session_meta = combined
            .iter()
            .find_map(|item| match item {
                RolloutItem::SessionMeta(meta_line) => Some(&meta_line.meta),
                _ => None,
            })
            .expect("session meta");

        assert_eq!(
            session_meta.merged_from_thread_ids,
            Some(vec![source_id, nested_id])
        );
    }

    #[tokio::test]
    async fn combine_appends_grandchild_full_history() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: grandchild_id,
                path: grandchild_path,
            }],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("g1".to_string(), Some("grandchild answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_preserves_child_and_grandchild_histories() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[
                CombineSource {
                    thread_id: child_id,
                    path: child_path,
                },
                CombineSource {
                    thread_id: grandchild_id,
                    path: grandchild_path,
                },
            ],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("g1".to_string(), Some("grandchild answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_preserves_selected_order_even_if_reversed() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[
                CombineSource {
                    thread_id: grandchild_id,
                    path: grandchild_path,
                },
                CombineSource {
                    thread_id: child_id,
                    path: child_path,
                },
            ],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("b1".to_string(), Some("a1".to_string())),
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
                ("g1".to_string(), Some("grandchild answer".to_string())),
                ("b1".to_string(), Some("a1".to_string())),
                ("c1".to_string(), Some("child answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_siblings_preserve_user_order_without_interleaving() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[
                CombineSource {
                    thread_id: first_id,
                    path: first_path,
                },
                CombineSource {
                    thread_id: second_id,
                    path: second_path,
                },
            ],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base".to_string(), Some("base answer".to_string())),
                ("base".to_string(), Some("base answer".to_string())),
                ("first".to_string(), Some("first answer".to_string())),
                ("base".to_string(), Some("base answer".to_string())),
                ("second".to_string(), Some("second answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_uses_logical_state_after_compaction() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base".to_string(), Some("base answer".to_string())),
                ("summary".to_string(), None),
                ("branch".to_string(), Some("branch answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_compacted_source_before_sibling_keeps_both_histories() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[
                CombineSource {
                    thread_id: compacted_id,
                    path: compacted_path,
                },
                CombineSource {
                    thread_id: sibling_id,
                    path: sibling_path,
                },
            ],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base".to_string(), Some("base answer".to_string())),
                ("summary".to_string(), None),
                ("first".to_string(), Some("first answer".to_string())),
                ("base".to_string(), Some("base answer".to_string())),
                ("second".to_string(), Some("second answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_preserves_rollback_with_duplicate_inherited_turns() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base-1".to_string(), Some("base-1 answer".to_string())),
                ("base-2".to_string(), Some("base-2 answer".to_string())),
                ("base-1".to_string(), Some("base-1 answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_preserves_rollback_before_new_turn() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base-1".to_string(), Some("base-1 answer".to_string())),
                ("base-2".to_string(), Some("base-2 answer".to_string())),
                ("base-1".to_string(), Some("base-1 answer".to_string())),
                ("branch".to_string(), Some("branch answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_uses_logical_state_after_rollback() {
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

        let merged = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: branch_id,
                path: branch_path,
            }],
        )
        .await
        .expect("combine");

        assert_eq!(
            logical_turn_pairs(&merged),
            vec![
                ("base".to_string(), Some("base answer".to_string())),
                ("base".to_string(), Some("base answer".to_string())),
            ]
        );
    }

    #[tokio::test]
    async fn combine_duplicate_source_ids_in_single_request_errors() {
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

        let err = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[
                CombineSource {
                    thread_id: child_id,
                    path: child_path.clone(),
                },
                CombineSource {
                    thread_id: child_id,
                    path: child_path,
                },
            ],
        )
        .await
        .expect_err("combine should fail");

        assert_eq!(
            err.to_string(),
            format!("duplicate combine source id {child_id}")
        );
    }

    #[tokio::test]
    async fn combine_source_id_must_match_rollout_session_id() {
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
        let actual_source_id = thread_id();
        let source_path = write_rollout(
            temp.path(),
            actual_source_id,
            None,
            &[TestTurn {
                user: "source",
                assistant: "source answer",
            }],
        );
        let mismatched_source_id = thread_id();

        let err = build_combined_rollout(
            base_id,
            base_path.as_path(),
            &[CombineSource {
                thread_id: mismatched_source_id,
                path: source_path.clone(),
            }],
        )
        .await
        .expect_err("combine should fail");

        assert_eq!(
            err.to_string(),
            format!(
                "combine source id {mismatched_source_id} does not match rollout session id {actual_source_id} for `{}`",
                source_path.display()
            )
        );
    }
}
