use anyhow::Result;
use app_test_support::McpProcess;
use app_test_support::create_fake_rollout;
use app_test_support::rollout_path;
use app_test_support::to_response;
use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::ThreadMergeParams;
use codex_app_server_protocol::ThreadMergeResponse;
use codex_app_server_protocol::ThreadReadParams;
use codex_app_server_protocol::ThreadReadResponse;
use codex_app_server_protocol::UserInput;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AgentMessageEvent;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::RolloutItem;
use codex_protocol::protocol::RolloutLine;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::SessionSource;
use codex_protocol::protocol::ThreadRolledBackEvent;
use codex_protocol::protocol::UserMessageEvent;
use pretty_assertions::assert_eq;
use serde_json::Value;
use std::path::Path;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::time::timeout;
use uuid::Uuid;

const DEFAULT_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

fn set_fork_parent(
    codex_home: &std::path::Path,
    filename_ts: &str,
    thread_id: &str,
    parent: &str,
) -> Result<()> {
    let path = rollout_path(codex_home, filename_ts, thread_id);
    let text = std::fs::read_to_string(&path)?;
    let mut lines: Vec<Value> = text
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<_, _>>()?;
    lines[0]["payload"]["forked_from_id"] = Value::String(parent.to_string());
    std::fs::write(
        &path,
        lines
            .into_iter()
            .map(|line| serde_json::to_string(&line))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
            + "\n",
    )?;
    Ok(())
}

fn create_config_toml(codex_home: &Path, server_uri: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(codex_home)?;
    let config = format!(
        "model_provider = \"mock_provider\"\n\
         [model_providers.mock_provider]\n\
         name = \"Mock Provider\"\n\
         base_url = \"{server_uri}/v1\"\n\
         wire_api = \"responses\"\n\
         env_key = \"DUMMY_KEY\"\n"
    );
    std::fs::write(codex_home.join("config.toml"), config)
}

fn create_rollout_with_turns(
    codex_home: &Path,
    filename_ts: &str,
    meta_rfc3339: &str,
    forked_from_id: Option<&str>,
    turns: &[(&str, &str)],
) -> Result<String> {
    let thread_id = Uuid::new_v4().to_string();
    let conversation_id = ThreadId::from_string(&thread_id)?;
    let file_path = rollout_path(codex_home, filename_ts, &thread_id);
    let dir = file_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("missing rollout parent directory"))?;
    std::fs::create_dir_all(dir)?;

    let mut lines = vec![RolloutLine {
        timestamp: meta_rfc3339.to_string(),
        item: codex_protocol::protocol::RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta {
                id: conversation_id,
                forked_from_id: forked_from_id.map(ThreadId::from_string).transpose()?,
                merge_base_thread_id: None,
                merged_from_thread_ids: None,
                timestamp: meta_rfc3339.to_string(),
                cwd: PathBuf::from("/"),
                originator: "codex".to_string(),
                cli_version: "0.0.0".to_string(),
                source: SessionSource::Cli,
                agent_nickname: None,
                agent_role: None,
                model_provider: Some("mock_provider".to_string()),
                base_instructions: None,
                dynamic_tools: None,
                memory_mode: None,
            },
            git: None,
        }),
    }];

    for (user, assistant) in turns {
        lines.push(RolloutLine {
            timestamp: meta_rfc3339.to_string(),
            item: codex_protocol::protocol::RolloutItem::ResponseItem(ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: (*user).to_string(),
                }],
                end_turn: None,
                phase: None,
            }),
        });
        lines.push(RolloutLine {
            timestamp: meta_rfc3339.to_string(),
            item: codex_protocol::protocol::RolloutItem::ResponseItem(ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: (*assistant).to_string(),
                }],
                end_turn: None,
                phase: None,
            }),
        });
        lines.push(RolloutLine {
            timestamp: meta_rfc3339.to_string(),
            item: codex_protocol::protocol::RolloutItem::EventMsg(EventMsg::UserMessage(
                UserMessageEvent {
                    message: (*user).to_string(),
                    images: Some(Vec::new()),
                    local_images: Vec::new(),
                    text_elements: Vec::new(),
                },
            )),
        });
        lines.push(RolloutLine {
            timestamp: meta_rfc3339.to_string(),
            item: codex_protocol::protocol::RolloutItem::EventMsg(EventMsg::AgentMessage(
                AgentMessageEvent {
                    message: (*assistant).to_string(),
                    phase: None,
                },
            )),
        });
    }

    std::fs::write(
        &file_path,
        lines
            .into_iter()
            .map(|line| serde_json::to_string(&line))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
            + "\n",
    )?;

    Ok(thread_id)
}

fn append_rollout_item(
    codex_home: &Path,
    filename_ts: &str,
    thread_id: &str,
    item: RolloutItem,
) -> Result<()> {
    let file_path = rollout_path(codex_home, filename_ts, thread_id);
    let text = std::fs::read_to_string(&file_path)?;
    let mut lines: Vec<RolloutLine> = text
        .lines()
        .map(serde_json::from_str::<RolloutLine>)
        .collect::<Result<_, _>>()?;
    lines.push(RolloutLine {
        timestamp: "2025-01-05T12:59:00Z".to_string(),
        item,
    });
    std::fs::write(
        &file_path,
        lines
            .into_iter()
            .map(|line| serde_json::to_string(&line))
            .collect::<Result<Vec<_>, _>>()?
            .join("\n")
            + "\n",
    )?;
    Ok(())
}

#[tokio::test]
async fn thread_merge_creates_new_thread_and_emits_started() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        "base",
        Some("mock_provider"),
        None,
    )?;
    let branch_a = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        "a",
        Some("mock_provider"),
        None,
    )?;
    let branch_b = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-02-00",
        "2025-01-05T12:02:00Z",
        "b",
        Some("mock_provider"),
        None,
    )?;
    set_fork_parent(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &branch_a,
        &base_id,
    )?;
    set_fork_parent(
        codex_home.path(),
        "2025-01-05T12-02-00",
        &branch_b,
        &base_id,
    )?;

    let base_path = rollout_path(codex_home.path(), "2025-01-05T12-00-00", &base_id);
    let base_before = std::fs::read_to_string(&base_path)?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id.clone(),
            merge_thread_ids: vec![branch_a.clone(), branch_b.clone()],
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadMergeResponse { thread, .. } = to_response::<ThreadMergeResponse>(response)?;

    assert_ne!(thread.id, base_id);
    assert_eq!(std::fs::read_to_string(&base_path)?, base_before);
    assert!(thread.path.is_some());

    Ok(())
}

#[tokio::test]
async fn thread_merge_rejects_non_descendants() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        "base",
        Some("mock_provider"),
        None,
    )?;
    let other_id = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        "other",
        Some("mock_provider"),
        None,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id,
            merge_thread_ids: vec![other_id],
            ..Default::default()
        })
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;
    assert!(error.error.message.contains("is not a descendant"));

    Ok(())
}

#[tokio::test]
async fn thread_merge_rejects_duplicate_source_ids() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        "base",
        Some("mock_provider"),
        None,
    )?;
    let child_id = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        "child",
        Some("mock_provider"),
        None,
    )?;
    set_fork_parent(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &child_id,
        &base_id,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id,
            merge_thread_ids: vec![child_id.clone(), child_id],
            ..Default::default()
        })
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;
    assert!(error.error.message.contains("duplicate merge thread id"));

    Ok(())
}

#[tokio::test]
async fn thread_merge_rejects_cyclic_descendants() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_fake_rollout(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        "base",
        Some("mock_provider"),
        None,
    )?;
    let cyclic_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        Some(&base_id),
        &[],
    )?;
    set_fork_parent(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &cyclic_id,
        &cyclic_id,
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id,
            merge_thread_ids: vec![cyclic_id],
            ..Default::default()
        })
        .await?;
    let error: JSONRPCError = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_error_message(RequestId::Integer(request_id)),
    )
    .await??;
    assert!(error.error.message.contains("is not a descendant"));

    Ok(())
}

#[tokio::test]
async fn thread_merge_grandchild_merge_normalizes_ancestor_order() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        None,
        &[],
    )?;
    let child_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        Some(&base_id),
        &[("child user", "child answer")],
    )?;
    let grandchild_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-02-00",
        "2025-01-05T12:02:00Z",
        Some(&child_id),
        &[
            ("child user", "child answer"),
            ("grandchild user", "grandchild answer"),
        ],
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id.clone(),
            merge_thread_ids: vec![grandchild_id, child_id],
            ephemeral: true,
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadMergeResponse { thread, .. } = to_response::<ThreadMergeResponse>(response)?;

    assert_ne!(thread.id, base_id);
    assert_eq!(
        thread.turns.len(),
        2,
        "expected child and grandchild suffix turns"
    );

    match &thread.turns[0].items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: "child user".to_string(),
                    text_elements: Vec::new(),
                }]
            );
        }
        other => panic!("expected first merged user message, got {other:?}"),
    }
    match &thread.turns[1].items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: "grandchild user".to_string(),
                    text_elements: Vec::new(),
                }]
            );
        }
        other => panic!("expected second merged user message, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn thread_merge_ephemeral_returns_merged_preview_and_turns() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        None,
        &[],
    )?;
    let branch_a = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        Some(&base_id),
        &[("branch a user", "branch a answer")],
    )?;
    let branch_b = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-02-00",
        "2025-01-05T12:02:00Z",
        Some(&base_id),
        &[("branch b user", "branch b answer")],
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id.clone(),
            merge_thread_ids: vec![branch_a, branch_b],
            ephemeral: true,
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadMergeResponse { thread, .. } = to_response::<ThreadMergeResponse>(response)?;

    assert_ne!(thread.id, base_id);
    assert!(
        thread.ephemeral,
        "ephemeral merges should be marked explicitly"
    );
    assert_eq!(thread.path, None, "ephemeral merges should remain pathless");
    assert_eq!(thread.preview, "branch a user");
    assert_eq!(thread.turns.len(), 2, "expected merged descendant turns");

    match &thread.turns[0].items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: "branch a user".to_string(),
                    text_elements: Vec::new(),
                }]
            );
        }
        other => panic!("expected first merged user message, got {other:?}"),
    }
    match &thread.turns[1].items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: "branch b user".to_string(),
                    text_elements: Vec::new(),
                }]
            );
        }
        other => panic!("expected second merged user message, got {other:?}"),
    }

    Ok(())
}

#[tokio::test]
async fn thread_merge_preserves_descendant_rollback_of_inherited_turn() -> Result<()> {
    let server = app_test_support::create_mock_responses_server_repeating_assistant("Done").await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri())?;

    let base_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-00-00",
        "2025-01-05T12:00:00Z",
        None,
        &[("base-1", "base-1 answer"), ("base-2", "base-2 answer")],
    )?;
    let branch_id = create_rollout_with_turns(
        codex_home.path(),
        "2025-01-05T12-01-00",
        "2025-01-05T12:01:00Z",
        Some(&base_id),
        &[("base-1", "base-1 answer"), ("base-2", "base-2 answer")],
    )?;
    append_rollout_item(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &branch_id,
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(ThreadRolledBackEvent {
            num_turns: 1,
        })),
    )?;
    append_rollout_item(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &branch_id,
        RolloutItem::ResponseItem(ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "branch".to_string(),
            }],
            end_turn: None,
            phase: None,
        }),
    )?;
    append_rollout_item(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &branch_id,
        RolloutItem::ResponseItem(ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: "branch answer".to_string(),
            }],
            end_turn: None,
            phase: None,
        }),
    )?;
    append_rollout_item(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &branch_id,
        RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
            message: "branch".to_string(),
            images: Some(Vec::new()),
            local_images: Vec::new(),
            text_elements: Vec::new(),
        })),
    )?;
    append_rollout_item(
        codex_home.path(),
        "2025-01-05T12-01-00",
        &branch_id,
        RolloutItem::EventMsg(EventMsg::AgentMessage(AgentMessageEvent {
            message: "branch answer".to_string(),
            phase: None,
        })),
    )?;

    let mut mcp = McpProcess::new(codex_home.path()).await?;
    timeout(DEFAULT_READ_TIMEOUT, mcp.initialize()).await??;

    let request_id = mcp
        .send_thread_merge_request(ThreadMergeParams {
            base_thread_id: base_id,
            merge_thread_ids: vec![branch_id],
            ..Default::default()
        })
        .await?;
    let response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(request_id)),
    )
    .await??;
    let ThreadMergeResponse { thread, .. } = to_response::<ThreadMergeResponse>(response)?;
    let read_request_id = mcp
        .send_thread_read_request(ThreadReadParams {
            thread_id: thread.id.clone(),
            include_turns: true,
        })
        .await?;
    let read_response: JSONRPCResponse = timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_response_message(RequestId::Integer(read_request_id)),
    )
    .await??;
    let ThreadReadResponse { thread } = to_response::<ThreadReadResponse>(read_response)?;

    assert_eq!(
        thread.turns.len(),
        2,
        "merged thread should expose the rolled-back final state"
    );
    match &thread.turns[0].items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: "base-1".to_string(),
                    text_elements: Vec::new(),
                }]
            );
        }
        other => panic!("expected first merged user message, got {other:?}"),
    }
    match &thread.turns[1].items[0] {
        ThreadItem::UserMessage { content, .. } => {
            assert_eq!(
                content,
                &vec![UserInput::Text {
                    text: "branch".to_string(),
                    text_elements: Vec::new(),
                }]
            );
        }
        other => panic!("expected second merged user message, got {other:?}"),
    }

    Ok(())
}
