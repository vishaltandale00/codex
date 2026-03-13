use crate::RolloutRecorder;
use crate::error::CodexErr;
use crate::error::Result as CodexResult;
use crate::find_thread_path_by_id_str;
use crate::parse_turn_item;
use crate::protocol::MergeBoundaryItem;
use crate::protocol::RolloutItem;
use crate::read_session_meta_line;
use codex_protocol::ThreadId;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::InitialHistory;
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

#[derive(Debug)]
struct ProvenanceBlock {
    source_thread_id: ThreadId,
    items: Vec<RolloutItem>,
}

pub async fn build_merged_rollout(
    base_thread_id: ThreadId,
    base_path: &Path,
    merge_sources: &[MergeSource],
) -> CodexResult<Vec<RolloutItem>> {
    let mut merged_items = load_rollout_items(base_path).await?;
    let mut covered_source_ids = covered_source_ids(&merged_items);
    let merged_source_ids: Vec<ThreadId> = merge_sources
        .iter()
        .map(|source| source.thread_id)
        .collect();
    let Some(RolloutItem::SessionMeta(meta_line)) = merged_items.first_mut() else {
        return Err(CodexErr::InvalidRequest(format!(
            "missing session metadata in base rollout `{}`",
            base_path.display()
        )));
    };
    meta_line.meta.merge_base_thread_id = Some(base_thread_id);
    meta_line.meta.merged_from_thread_ids = Some(merged_source_ids);

    let ordered_sources = order_merge_sources(base_path, merge_sources).await?;
    for source in ordered_sources {
        let source_items = load_rollout_items(source.path.as_path()).await?;
        let merged_turns = effective_turns(&merged_items);
        let source_turns = effective_turns(&source_items);
        let inherited_turn_count = longest_common_prefix_len(&merged_turns, &source_turns);
        if inherited_turn_count >= source_turns.len() {
            continue;
        }

        let cut_idx = source_turns[inherited_turn_count].start_idx;
        for block in suffix_blocks(source.thread_id, source_items, cut_idx) {
            if !covered_source_ids.insert(block.source_thread_id) {
                continue;
            }
            merged_items.push(RolloutItem::MergeBoundary(MergeBoundaryItem {
                source_thread_id: block.source_thread_id,
            }));
            merged_items.extend(block.items);
        }
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

    for (idx, item) in items.iter().enumerate() {
        match item {
            RolloutItem::ResponseItem(response_item) => {
                if matches!(
                    parse_turn_item(response_item),
                    Some(TurnItem::UserMessage(_))
                ) {
                    if let Some(turn) = active_turn.take() {
                        turns.push(turn);
                    }
                    active_turn = Some(EffectiveTurn {
                        start_idx: idx,
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
                let new_len = turns.len().saturating_sub(num_turns);
                turns.truncate(new_len);
            }
            _ => {}
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

fn covered_source_ids(items: &[RolloutItem]) -> HashSet<ThreadId> {
    items
        .iter()
        .filter_map(|item| match item {
            RolloutItem::MergeBoundary(boundary) => Some(boundary.source_thread_id),
            _ => None,
        })
        .collect()
}

fn suffix_blocks(
    default_source_thread_id: ThreadId,
    source_items: Vec<RolloutItem>,
    cut_idx: usize,
) -> Vec<ProvenanceBlock> {
    let mut blocks = Vec::new();
    let mut current_source_thread_id = default_source_thread_id;
    let mut current_items = Vec::new();

    for item in source_items.into_iter().skip(cut_idx) {
        match item {
            RolloutItem::MergeBoundary(boundary) => {
                if !current_items.is_empty() {
                    blocks.push(ProvenanceBlock {
                        source_thread_id: current_source_thread_id,
                        items: current_items,
                    });
                    current_items = Vec::new();
                }
                current_source_thread_id = boundary.source_thread_id;
            }
            _ => current_items.push(item),
        }
    }

    if !current_items.is_empty() {
        blocks.push(ProvenanceBlock {
            source_thread_id: current_source_thread_id,
            items: current_items,
        });
    }

    blocks
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
    let mut current_path = descendant.path.clone();
    let mut seen = HashSet::new();

    loop {
        let meta_line = read_session_meta_line(current_path.as_path()).await?;
        let Some(parent_thread_id) = meta_line.meta.forked_from_id else {
            return Ok(false);
        };
        if parent_thread_id == ancestor.thread_id {
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
    use codex_protocol::ThreadId;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
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
                    rollout_line_message("user", turn.user).item,
                    rollout_line_message("assistant", turn.assistant).item,
                ]
            })
            .collect()
    }

    fn rollout_line_message(role: &str, text: &str) -> RolloutLine {
        RolloutLine {
            timestamp: "2026-01-01T00:00:00.000Z".to_string(),
            item: RolloutItem::ResponseItem(ResponseItem::Message {
                id: None,
                role: role.to_string(),
                content: vec![ContentItem::InputText {
                    text: text.to_string(),
                }],
                end_turn: None,
                phase: None,
            }),
        }
    }

    fn response_texts(items: &[RolloutItem], role: &str) -> Vec<String> {
        items
            .iter()
            .filter_map(|item| match item {
                RolloutItem::ResponseItem(ResponseItem::Message {
                    role: item_role,
                    content,
                    ..
                }) if item_role == role => {
                    content.iter().find_map(|content_item| match content_item {
                        ContentItem::InputText { text } => Some(text.clone()),
                        _ => None,
                    })
                }
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn merge_rollout_preserves_divergent_turns_with_identical_user_prompts() {
        let temp = TempDir::new().expect("tempdir");
        let base_id = thread_id();
        let base_path = write_rollout(
            temp.path(),
            base_id,
            None,
            &[TestTurn {
                user: "shared prompt",
                assistant: "base answer",
            }],
        );
        let branch_id = thread_id();
        let branch_path = write_rollout(
            temp.path(),
            branch_id,
            Some(base_id),
            &[TestTurn {
                user: "shared prompt",
                assistant: "branch answer",
            }],
        );

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
            response_texts(&merged, "assistant"),
            vec!["base answer".to_string(), "branch answer".to_string()]
        );
        assert!(merged.iter().any(|item| {
            matches!(
                item,
                RolloutItem::MergeBoundary(MergeBoundaryItem { source_thread_id })
                if *source_thread_id == branch_id
            )
        }));
    }

    #[tokio::test]
    async fn merge_rollout_appends_only_incremental_descendant_suffixes() {
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
        let grandchild_id = thread_id();
        let grandchild_path = write_rollout(
            temp.path(),
            grandchild_id,
            Some(child_id),
            &[
                TestTurn {
                    user: "base",
                    assistant: "base answer",
                },
                TestTurn {
                    user: "child",
                    assistant: "child answer",
                },
                TestTurn {
                    user: "grandchild",
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
            response_texts(&merged, "assistant"),
            vec![
                "base answer".to_string(),
                "child answer".to_string(),
                "grandchild answer".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn merge_rollout_orders_ancestors_before_descendants_even_when_reversed() {
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
        let grandchild_id = thread_id();
        let grandchild_path = write_rollout(
            temp.path(),
            grandchild_id,
            Some(child_id),
            &[
                TestTurn {
                    user: "base",
                    assistant: "base answer",
                },
                TestTurn {
                    user: "child",
                    assistant: "child answer",
                },
                TestTurn {
                    user: "grandchild",
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
            response_texts(&merged, "assistant"),
            vec![
                "base answer".to_string(),
                "child answer".to_string(),
                "grandchild answer".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn merge_rollout_uses_rollback_adjusted_positions() {
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
        let branch_path = write_rollout(
            temp.path(),
            branch_id,
            Some(base_id),
            &[
                TestTurn {
                    user: "base-1",
                    assistant: "base-1 answer",
                },
                TestTurn {
                    user: "base-2",
                    assistant: "base-2 answer",
                },
                TestTurn {
                    user: "branch-1",
                    assistant: "branch-1 answer",
                },
            ],
        );
        let mut items = crate::RolloutRecorder::get_rollout_history(branch_path.as_path())
            .await
            .expect("history")
            .get_rollout_items();
        items.push(RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            ThreadRolledBackEvent { num_turns: 1 },
        )));
        let lines = items
            .into_iter()
            .map(|item| RolloutLine {
                timestamp: "2026-01-01T00:00:00.000Z".to_string(),
                item,
            })
            .map(|line| serde_json::to_string(&line).expect("serialize"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&branch_path, lines).expect("rewrite rollout");

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
            response_texts(&merged, "assistant"),
            vec!["base-1 answer".to_string(), "base-2 answer".to_string()]
        );
    }

    #[tokio::test]
    async fn merge_rollout_skips_duplicate_blocks_from_prior_merged_descendant() {
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
        let a_id = thread_id();
        let a_turn = TestTurn {
            user: "a",
            assistant: "a answer",
        };
        write_rollout(temp.path(), a_id, Some(base_id), &[a_turn]);
        let c_id = thread_id();
        let c_path = write_rollout(
            temp.path(),
            c_id,
            Some(base_id),
            &[TestTurn {
                user: "c",
                assistant: "c answer",
            }],
        );
        let merged_ac_id = thread_id();
        let merged_ac_path = write_rollout_items(
            temp.path(),
            merged_ac_id,
            Some(base_id),
            vec![
                RolloutItem::MergeBoundary(MergeBoundaryItem {
                    source_thread_id: a_id,
                }),
                rollout_line_message("user", "a").item,
                rollout_line_message("assistant", "a answer").item,
                RolloutItem::MergeBoundary(MergeBoundaryItem {
                    source_thread_id: c_id,
                }),
                rollout_line_message("user", "c").item,
                rollout_line_message("assistant", "c answer").item,
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: c_id,
                    path: c_path,
                },
                MergeSource {
                    thread_id: merged_ac_id,
                    path: merged_ac_path,
                },
            ],
        )
        .await
        .expect("merge");

        assert_eq!(
            response_texts(&merged, "assistant"),
            vec![
                "base answer".to_string(),
                "c answer".to_string(),
                "a answer".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn merge_rollout_preserves_requested_order_when_deduping_prior_merge() {
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
        let a_id = thread_id();
        write_rollout(
            temp.path(),
            a_id,
            Some(base_id),
            &[TestTurn {
                user: "a",
                assistant: "a answer",
            }],
        );
        let c_id = thread_id();
        let c_path = write_rollout(
            temp.path(),
            c_id,
            Some(base_id),
            &[TestTurn {
                user: "c",
                assistant: "c answer",
            }],
        );
        let merged_ac_id = thread_id();
        let merged_ac_path = write_rollout_items(
            temp.path(),
            merged_ac_id,
            Some(base_id),
            vec![
                RolloutItem::MergeBoundary(MergeBoundaryItem {
                    source_thread_id: a_id,
                }),
                rollout_line_message("user", "a").item,
                rollout_line_message("assistant", "a answer").item,
                RolloutItem::MergeBoundary(MergeBoundaryItem {
                    source_thread_id: c_id,
                }),
                rollout_line_message("user", "c").item,
                rollout_line_message("assistant", "c answer").item,
            ],
        );

        let merged = build_merged_rollout(
            base_id,
            base_path.as_path(),
            &[
                MergeSource {
                    thread_id: merged_ac_id,
                    path: merged_ac_path,
                },
                MergeSource {
                    thread_id: c_id,
                    path: c_path,
                },
            ],
        )
        .await
        .expect("merge");

        assert_eq!(
            response_texts(&merged, "assistant"),
            vec![
                "base answer".to_string(),
                "a answer".to_string(),
                "c answer".to_string(),
            ]
        );
    }
}
