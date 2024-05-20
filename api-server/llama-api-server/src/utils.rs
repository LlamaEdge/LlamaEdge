pub(crate) fn log(msg: impl std::fmt::Display) {
    println!("{}", msg);
}

pub(crate) fn gen_chat_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}
