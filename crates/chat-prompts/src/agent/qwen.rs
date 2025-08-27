use crate::{
    error::{PromptError, Result},
    BuildChatPrompt,
};
use endpoints::chat::{
    ChatCompletionAssistantMessage, ChatCompletionRequestMessage, ChatCompletionSystemMessage,
    ChatCompletionToolMessage, ChatCompletionUserMessage, ChatCompletionUserMessageContent,
    ContentPart, Tool,
};

const QWEN3_AGENT_SYSTEM_PROMPT: &str = r#"
# 角色设定
你是一个严格遵守规则的智能助手，负责解析用户输入，并在必要时调用工具。你必须在回答中严格遵守以下规范。

---

## 工具调用准则
- 工具白名单（严格匹配）：
  - 你只能调用【本次任务可用工具】部分列出的工具。调用前必须在 <thought> 中确认所选工具名“完全等于”白名单之一。
  - 如果没有任何可用工具适用，必须直接输出 <final_answer>，绝不编造工具名或参数。
- 常识 vs 动态规则：
  - 若用户问题涉及**常识性、稳定知识**（如数学公式、编程语法、历史事实），即使有工具，也应直接在 <final_answer> 中回答，不必调用工具。
  - 若用户问题涉及**动态性、实时性或超出静态知识范围**（如天气、新闻、股市、当前数据），且提供了相关工具（如 web search），则必须调用工具，不得凭空回答。

---

## 输出步骤
1. <question>：原样复述用户输入
2. <thought>：详细说明你的推理过程，包括是否需要工具、为什么选择某个工具、工具参数构造方式
3. <action>：如果需要调用工具，输出 JSON 格式的工具调用，立即停止生成
4. <observation>：填入系统返回的工具调用结果
5. <final_answer>：基于已有信息或工具返回，输出最终答案

---

## 格式要求
- 必须严格使用以下 XML 标签：
  - <question> 用户问题
  - <thought> 思考
  - <action> 工具调用
  - <observation> 工具返回结果
  - <final_answer> 最终答案
- 每次回答必须包含两个标签：先 <thought>，再 <action> 或 <final_answer>
- 工具调用必须使用 JSON 格式，例如：
  <action>{"name": "get_height", "arguments": {"query": "埃菲尔铁塔"}}</action>
- 输出 <action> 后立即停止生成，等待 <observation>
- 文件路径必须是绝对路径；多行参数用 \n 表示
- **回答语言必须与用户输入语言保持一致**（例如用户用英文提问，就用英文回答；用户用中文提问，就用中文回答）
- `<final_answer>` 标签必须包含完整的最终答案，包括文字说明和代码。
- 代码示例必须写在 `<final_answer>` 内部，使用 Markdown ``` 语法包裹，不得出现在 `<final_answer>` 之外。
- 不得在 `<final_answer>` 外部输出额外的解释、代码或答案。

---

## 本次任务可用工具
${tool_list}

---

## 示例

### 示例 1：常识问题（无需工具，英文）
用户输入：
What is the capital of France?

模型输出：
<question>What is the capital of France?</question>
<thought>This is a stable knowledge question (the capital of France does not change often). According to the "common sense vs dynamic" rule, I should answer directly without using any tool.</thought>
<final_answer>The capital of France is Paris.</final_answer>

---

### 示例 2：动态问题（必须调用工具，英文）
用户输入：
What's the weather in Singapore now?

模型输出：
<question>What's the weather in Singapore now?</question>
<thought>This is a dynamic query that changes over time. According to the rules, I must use the provided web search tool to get the latest information.</thought>
<action>{"name": "web_search", "arguments": {"query": "current weather in Singapore"}}</action>

（系统返回结果后，模型继续输出）
<observation>{"temperature": "30°C", "condition": "Partly cloudy"}</observation>
<final_answer>The current weather in Singapore is about 30°C and partly cloudy.</final_answer>

---

### 示例 3：常识问题（无需工具，中文）
用户输入：
法国的首都是哪里？

模型输出：
<question>法国的首都是哪里？</question>
<thought>这是一个稳定的常识性问题（法国的首都不会频繁变化）。根据“常识 vs 动态”规则，我应该直接回答，不需要调用工具。</thought>
<final_answer>法国的首都是巴黎。</final_answer>

---

### 示例 4：动态问题（必须调用工具，中文）
用户输入：
现在新加坡的天气怎么样？

模型输出：
<question>现在新加坡的天气怎么样？</question>
<thought>这是一个随时间变化的动态性问题。根据规则，我必须调用提供的 web_search 工具来获取最新信息。</thought>
<action>{"name": "web_search", "arguments": {"query": "新加坡当前天气"}}</action>

（系统返回结果后，模型继续输出）
<observation>{"temperature": "30°C", "condition": "多云"}</observation>
<final_answer>新加坡现在的天气大约是 30°C，多云。</final_answer>

---

### 示例 5：语言一致性边界案例（中英混合）
用户输入：
请用英文告诉我法国的首都是哪里

模型输出：
<question>请用英文告诉我法国的首都是哪里</question>
<thought>用户明确要求用英文回答。虽然问题本身是中文输入，但回答语言必须遵循用户指令，保持语言一致性。因此最终答案应为英文。</thought>
<final_answer>The capital of France is Paris.</final_answer>

---

### 示例 6：语言一致性边界案例（用户输入英文，但要求输出中文）
用户输入：
Please answer in Chinese: What is the capital of France?

模型输出：
<question>Please answer in Chinese: What is the capital of France?</question>
<thought>虽然用户输入是英文，但明确要求用中文回答。根据语言一致性规则，最终答案必须遵循用户的指令，使用中文输出。</thought>
<final_answer>法国的首都是巴黎。</final_answer>
"#;

/// Generate prompts for the `Qwen3` models in tool use scenario.
#[derive(Debug, Default, Clone)]
pub struct Qwen3AgentPrompt;
impl Qwen3AgentPrompt {
    /// Create a system prompt from a chat completion request message.
    fn create_system_prompt(&self, message: &ChatCompletionSystemMessage) -> String {
        let content = message.content();
        match content.is_empty() {
            true => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible.<|im_end|>"),
            false => format!("<|im_start|>system\n{content}<|im_end|>"),
        }
    }

    /// Create a user prompt from a chat completion request message.
    fn append_user_message(
        &self,
        chat_history: impl AsRef<str>,
        system_prompt: impl AsRef<str>,
        message: &ChatCompletionUserMessage,
    ) -> String {
        let content = match message.content() {
            ChatCompletionUserMessageContent::Text(text) => text.to_string(),
            ChatCompletionUserMessageContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    if let ContentPart::Text(text_content) = part {
                        content.push_str(text_content.text());
                        content.push('\n');
                    }
                }
                content
            }
        };

        match chat_history.as_ref().is_empty() {
            true => match system_prompt.as_ref().is_empty() {
                true => {
                    format!(
                        "<|im_start|>user\n{user_message}<|im_end|>",
                        user_message = content.trim(),
                    )
                }
                false => {
                    format!(
                        "{system_prompt}\n<|im_start|>user\n{user_message}<|im_end|>",
                        system_prompt = system_prompt.as_ref().trim(),
                        user_message = content.trim(),
                    )
                }
            },
            false => format!(
                "{chat_history}\n<|im_start|>user\n{user_message}<|im_end|>",
                chat_history = chat_history.as_ref().trim(),
                user_message = content.trim(),
            ),
        }
    }

    /// create an assistant prompt from a chat completion request message.
    fn append_assistant_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionAssistantMessage,
    ) -> Result<String> {
        let content = match message.content() {
            Some(content) => content.to_string(),
            // Note that the content is optional if `tool_calls` is specified.
            None => match message.tool_calls() {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    let mut functions = vec![];
                    for tool_call in tool_calls {
                        functions.push(&tool_call.function);
                    }
                    serde_json::to_string(&functions).unwrap()
                }
                _ => return Err(PromptError::NoAssistantMessage),
            },
        };

        Ok(format!(
            "{chat_history}\n<|im_start|>assistant\n{assistant_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            assistant_message = content.trim(),
        ))
    }

    /// create a tool prompt from a chat completion request message.
    fn append_tool_message(
        &self,
        chat_history: impl AsRef<str>,
        message: &ChatCompletionToolMessage,
    ) -> String {
        format!(
            "{chat_history}\n<|im_start|>user\n{tool_message}<|im_end|>",
            chat_history = chat_history.as_ref().trim(),
            tool_message = message.content().trim(),
        )
    }
}
impl BuildChatPrompt for Qwen3AgentPrompt {
    fn build(&self, messages: &mut Vec<ChatCompletionRequestMessage>) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => self.create_system_prompt(message),
            _ => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible.<|im_end|>"),
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant\n<think>/n/n</think>");

        Ok(prompt)
    }

    fn build_with_tools(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        tools: Option<&[Tool]>,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(crate::error::PromptError::NoMessages);
        }

        // system prompt
        let system_prompt = match messages[0] {
            ChatCompletionRequestMessage::System(ref message) => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    let system_prompt = QWEN3_AGENT_SYSTEM_PROMPT.replace("${tool_list}", &available_tools);

                    format!("<|im_start|>system\n{system_prompt}\n<|im_end|>")
                }
                _ => self.create_system_prompt(message),
            },
            _ => match tools {
                Some(tools) if !tools.is_empty() => {
                    let available_tools = serde_json::to_string_pretty(tools).unwrap();

                    let system_prompt = QWEN3_AGENT_SYSTEM_PROMPT.replace("${tool_list}", &available_tools);

                    format!("<|im_start|>system\n{system_prompt}\n<|im_end|>")
                }
                _ => String::from("<|im_start|>system\nYou are a helpful assistant. Answer questions as concisely as possible.<|im_end|>"),
            },
        };

        // append user/assistant messages
        let mut prompt = String::new();
        for message in messages {
            match message {
                ChatCompletionRequestMessage::User(message) => {
                    prompt = self.append_user_message(&prompt, &system_prompt, message);
                }
                ChatCompletionRequestMessage::Assistant(message) => {
                    prompt = self.append_assistant_message(&prompt, message)?;
                }
                ChatCompletionRequestMessage::Tool(message) => {
                    prompt = self.append_tool_message(&prompt, message);
                }
                _ => continue,
            }
        }

        prompt.push_str("\n<|im_start|>assistant\n<think>/n/n</think>");

        Ok(prompt)
    }
}
