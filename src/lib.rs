use dotenv::dotenv;
use github_flows::{listen_to_event, EventPayload, GithubLogin::Provided};
use openai_flows::chat::{ChatModel, ChatOptions};
use openai_flows::{FlowsAccount, OpenAIFlows};
use slack_flows::send_message_to_channel;
use std::env;
use tiktoken_rs::cl100k_base;
#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn run() -> anyhow::Result<()> {
    dotenv().ok();
    let github_login = env::var("github_login").unwrap_or("alabulei1".to_string());
    let github_owner = env::var("github_owner").unwrap_or("alabulei1".to_string());
    let github_repo = env::var("github_repo").unwrap_or("a-test".to_string());

    listen_to_event(
        &Provided(github_login.clone()),
        &github_owner,
        &github_repo,
        vec!["issues"],
        handler,
    )
    .await;

    Ok(())
}

async fn handler(payload: EventPayload) {
    let openai_key_name = env::var("openai_key_name").unwrap_or("secondstate".to_string());
    let slack_workspace = env::var("slack_workspace").unwrap_or("secondstate".to_string());
    let slack_channel = env::var("slack_channel").unwrap_or("github-status".to_string());

    if let EventPayload::IssuesEvent(e) = payload {
        let issue = e.issue;
        let issue_title = issue.title;
        let issue_number = issue.number;
        let issue_body = issue.body.unwrap();
        let issue_url = issue.html_url;
        let labels = issue
            .labels
            .into_iter()
            .map(|lab| lab.name)
            .collect::<Vec<String>>()
            .join(", ");

        let bpe = cl100k_base().unwrap();
        let mut openai = OpenAIFlows::new();
        openai.set_flows_account(FlowsAccount::Provided(openai_key_name));
        openai.set_retry_times(1);
        let system = &format!("You are the co-owner of a github repo, you monitor new issues by analyzing the title, body text, labels and its context");

        let co = ChatOptions {
            model: ChatModel::GPT35Turbo,
            restart: true,
            system_prompt: Some(system),
        };
        let chat_id = format!("ISSUE#{issue_number}");

        let tokens = bpe.encode_ordinary(&issue_body);
        let total_tokens_count = tokens.len();
        let mut _summary = "";

        if total_tokens_count > 3000 {
            let mut token_vec = tokens.clone();
            let mut map_out = "".to_string();

            while !token_vec.is_empty() {
                let token_chunk = token_vec.drain(0..3000).collect::<Vec<_>>();

                let issue_body_chunk = bpe.decode(token_chunk).unwrap();

                let map_question = format!("The issue is titled {issue_title}, labeled {labels}, with one chunk of the body text {issue_body_chunk}. Please summarize key information in this section.");

                match openai.chat_completion(&chat_id, &map_question, &co).await {
                    Ok(r) => {
                        map_out.push_str(r.choice.trim());
                    }
                    Err(_e) => {}
                }
            }

            let reduce_question = format!("The issue is titled {issue_title}, with summarized key info of its chunks {map_out}, please make a concise summary for this issue to facilitate the next action.");

            match openai
                .chat_completion(&chat_id, &reduce_question, &co)
                .await
            {
                Ok(r) => {
                    _summary = r.choice.trim();
                    return;
                }
                Err(_e) => {}
            }
        } else {
            let question = format!("The issue is titled {issue_title}, labeled {labels}, with body text {issue_body}, based on this context, please make a concise summary for this issue to facilitate the next action.");

            match openai.chat_completion(&chat_id, &question, &co).await {
                Ok(r) => {
                    _summary = r.choice.trim();
                    return;
                }
                Err(_e) => {}
            }
        }

        let text = format!("New issue:\n{}\n{}", _summary, issue_url);
        send_message_to_channel(&slack_workspace, &slack_channel, text);
    }
}
