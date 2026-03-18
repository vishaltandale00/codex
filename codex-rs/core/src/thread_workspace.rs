use crate::config::Config;
use crate::git_info::resolve_root_git_project_for_trust;
use crate::path_utils::normalize_for_path_comparison;
use crate::rollout::list::read_session_meta_line;
use crate::state_db::get_state_db;
use codex_protocol::ThreadId;
use std::path::Path;
use std::path::PathBuf;
use tracing::warn;

pub async fn resolve_recorded_thread_cwd(
    config: &Config,
    thread_id: Option<ThreadId>,
    rollout_path: &Path,
) -> Option<PathBuf> {
    if let Some(state_db_ctx) = get_state_db(config).await
        && let Some(thread_id) = thread_id
        && let Ok(Some(metadata)) = state_db_ctx.get_thread(thread_id).await
        && !metadata.cwd.as_os_str().is_empty()
    {
        return Some(metadata.cwd);
    }

    match read_session_meta_line(rollout_path).await {
        Ok(meta_line) if !meta_line.meta.cwd.as_os_str().is_empty() => Some(meta_line.meta.cwd),
        Ok(_) => None,
        Err(err) => {
            let rollout_path = rollout_path.display();
            warn!("failed to read session metadata from rollout {rollout_path}: {err}");
            None
        }
    }
}

pub fn paths_share_workspace(lhs_cwd: &Path, rhs_cwd: &Path) -> bool {
    if paths_match(lhs_cwd, rhs_cwd) {
        return true;
    }

    match (
        resolve_root_git_project_for_trust(lhs_cwd),
        resolve_root_git_project_for_trust(rhs_cwd),
    ) {
        (Some(lhs_root), Some(rhs_root)) => paths_match(lhs_root.as_path(), rhs_root.as_path()),
        _ => false,
    }
}

fn paths_match(lhs: &Path, rhs: &Path) -> bool {
    if let (Ok(lhs), Ok(rhs)) = (
        normalize_for_path_comparison(lhs),
        normalize_for_path_comparison(rhs),
    ) {
        return lhs == rhs;
    }

    lhs == rhs
}

#[cfg(test)]
mod tests {
    use super::paths_share_workspace;
    use super::resolve_recorded_thread_cwd;
    use crate::config::ConfigBuilder;
    use chrono::DateTime;
    use chrono::Utc;
    use codex_protocol::ThreadId;
    use codex_protocol::protocol::RolloutItem;
    use codex_protocol::protocol::RolloutLine;
    use codex_protocol::protocol::SessionMeta;
    use codex_protocol::protocol::SessionMetaLine;
    use codex_protocol::protocol::SessionSource;
    use codex_state::StateRuntime;
    use codex_state::ThreadMetadataBuilder;
    use pretty_assertions::assert_eq;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use uuid::Uuid;

    fn write_rollout_with_session_meta(
        root: &Path,
        thread_id: ThreadId,
        cwd: PathBuf,
    ) -> std::io::Result<PathBuf> {
        let dir = root.join("sessions").join("2026").join("01").join("01");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("rollout-2026-01-01T00-00-00-{thread_id}.jsonl"));
        let line = RolloutLine {
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            item: RolloutItem::SessionMeta(SessionMetaLine {
                meta: SessionMeta {
                    id: thread_id,
                    forked_from_id: None,
                    merge_base_thread_id: None,
                    merged_from_thread_ids: None,
                    timestamp: "2026-01-01T00:00:00Z".to_string(),
                    cwd,
                    originator: "test".to_string(),
                    cli_version: "0.0.0".to_string(),
                    source: SessionSource::Cli,
                    agent_nickname: None,
                    agent_role: None,
                    model_provider: Some("openai".to_string()),
                    base_instructions: None,
                    dynamic_tools: None,
                    memory_mode: None,
                },
                git: None,
            }),
        };
        let json = serde_json::to_string(&line).expect("serialize rollout");
        let mut file = File::create(&path)?;
        writeln!(file, "{json}")?;
        Ok(path)
    }

    #[tokio::test]
    async fn resolve_recorded_thread_cwd_prefers_state_db_when_rollout_cwd_is_empty() {
        let temp = TempDir::new().expect("tempdir");
        let thread_id = ThreadId::from_string(&Uuid::new_v4().to_string()).expect("thread id");
        let rollout_path = write_rollout_with_session_meta(temp.path(), thread_id, PathBuf::new())
            .expect("write rollout");
        let expected_cwd = temp.path().join("workspace");
        std::fs::create_dir_all(&expected_cwd).expect("create workspace");

        let config = ConfigBuilder::default()
            .codex_home(temp.path().to_path_buf())
            .build()
            .await
            .expect("build config");
        let state_db =
            StateRuntime::init(config.sqlite_home.clone(), config.model_provider_id.clone())
                .await
                .expect("state db should initialize");
        state_db
            .mark_backfill_complete(None)
            .await
            .expect("backfill should complete");

        let created_at = DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .expect("timestamp")
            .with_timezone(&Utc);
        let mut builder = ThreadMetadataBuilder::new(
            thread_id,
            rollout_path.clone(),
            created_at,
            SessionSource::Cli,
        );
        builder.cwd = expected_cwd.clone();
        let metadata = builder.build(config.model_provider_id.as_str());
        state_db
            .upsert_thread(&metadata)
            .await
            .expect("upsert thread metadata");

        let cwd = resolve_recorded_thread_cwd(&config, Some(thread_id), rollout_path.as_path())
            .await
            .expect("resolved cwd");
        assert_eq!(cwd, expected_cwd);
    }

    #[tokio::test]
    async fn resolve_recorded_thread_cwd_uses_rollout_cwd_when_state_db_is_missing() {
        let temp = TempDir::new().expect("tempdir");
        let thread_id = ThreadId::from_string(&Uuid::new_v4().to_string()).expect("thread id");
        let expected_cwd = temp.path().join("workspace");
        std::fs::create_dir_all(&expected_cwd).expect("create workspace");
        let rollout_path =
            write_rollout_with_session_meta(temp.path(), thread_id, expected_cwd.clone())
                .expect("write rollout");

        let config = ConfigBuilder::default()
            .codex_home(temp.path().to_path_buf())
            .build()
            .await
            .expect("build config");

        let cwd = resolve_recorded_thread_cwd(&config, Some(thread_id), rollout_path.as_path())
            .await
            .expect("resolved cwd");
        assert_eq!(cwd, expected_cwd);
    }

    #[test]
    fn paths_share_workspace_matches_sibling_worktrees_in_same_repo() {
        let temp = TempDir::new().expect("tempdir");
        let repo_root = temp.path().join("repo");
        let workspace_a = repo_root.join("workspace-a");
        let workspace_b = repo_root.join("workspace-b");
        std::fs::create_dir_all(repo_root.join(".git")).expect("create .git");
        std::fs::create_dir_all(&workspace_a).expect("create workspace a");
        std::fs::create_dir_all(&workspace_b).expect("create workspace b");

        assert!(paths_share_workspace(
            workspace_a.as_path(),
            workspace_b.as_path()
        ));
    }

    #[test]
    fn paths_share_workspace_rejects_different_repo_roots() {
        let temp = TempDir::new().expect("tempdir");
        let repo_a = temp.path().join("repo-a");
        let repo_b = temp.path().join("repo-b");
        let workspace_a = repo_a.join("workspace");
        let workspace_b = repo_b.join("workspace");
        std::fs::create_dir_all(repo_a.join(".git")).expect("create repo a");
        std::fs::create_dir_all(repo_b.join(".git")).expect("create repo b");
        std::fs::create_dir_all(&workspace_a).expect("create workspace a");
        std::fs::create_dir_all(&workspace_b).expect("create workspace b");

        assert!(!paths_share_workspace(
            workspace_a.as_path(),
            workspace_b.as_path()
        ));
    }
}
