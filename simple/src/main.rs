use std::env;
use wasi_nn;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let prompt: &str = &args[2];

    let graph =
        wasi_nn::GraphBuilder::new(wasi_nn::GraphEncoding::Ggml, wasi_nn::ExecutionTarget::AUTO)
            .build_from_cache(model_name)
            .expect("Failed to load the model");

    let mut context = graph.init_execution_context().expect("Failed to init context");
    let tensor_data = prompt.as_bytes().to_vec();
    context
        .set_input(0, wasi_nn::TensorType::U8, &[1], &tensor_data)
        .expect("Failed to set prompt as the input tensor");
    context.compute().expect("Failed to complete inference");

    let max_output_size = 4096*6;
    let mut output_buffer = vec![0u8; max_output_size];
    let mut output_size = context.get_output(0, &mut output_buffer).expect("Failed to get inference output");
    output_size = std::cmp::min(max_output_size, output_size);
    let output = String::from_utf8_lossy(&output_buffer[..output_size]).to_string();

    println!("{}", prompt);
    println!("{}", output);
}
