use std::env;
use std::io;
use wasi_nn;

fn read_input() -> String {
    loop {
        let mut answer = String::new();
        io::stdin()
            .read_line(&mut answer)
            .ok()
            .expect("Failed to read line");
        if !answer.is_empty() && answer != "\n" && answer != "\r\n" {
            return answer;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(model_name)
            .unwrap();
    let mut context = graph.init_execution_context().unwrap();

    let system_prompt = String::from("<<SYS>>You are a helpful, respectful and honest assistant. Always answer as short as possible, while being safe. <</SYS>>");
    let mut saved_prompt = String::new();

    // Ask a quick question to load the model
    let initial_prompt = "Are you ready to answer questions? Answer yes or no.";
    context
        .set_input(
            0,
            wasi_nn::TensorType::U8,
            &[1],
            &initial_prompt.as_bytes().to_vec(),
        )
        .unwrap();
    context.compute().unwrap();

    loop {
        println!("Question:");
        let input = read_input();
        if saved_prompt == "" {
            saved_prompt = format!("[INST] {} {} [/INST]", system_prompt, input.trim());
        } else {
            saved_prompt = format!("{} [INST] {} [/INST]", saved_prompt, input.trim());
        }

        // Set prompt to the input tensor.
        let tensor_data = saved_prompt.as_bytes().to_vec();
        context
            .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
            .unwrap();

        // Execute the inference.
        context.compute().unwrap();

        // Retrieve the output.
        let max_output_size = 4096 * 6;
        let mut output_buffer = vec![0u8; max_output_size];
        let mut output_size = context.get_output(0, &mut output_buffer).unwrap();
        output_size = std::cmp::min(max_output_size, output_size);
        let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();
        println!("Answer:\n{}", output.trim());

        saved_prompt = format!("{} {} ", saved_prompt, output.trim());
    }
}
