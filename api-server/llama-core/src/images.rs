use crate::{error::LlamaCoreError, SD_IMAGE_TO_IMAGE, SD_TEXT_TO_IMAGE};
use base64::{engine::general_purpose, Engine as _};
use endpoints::images::{ImageCreateRequest, ImageEditRequest, ImageObject, ListImagesResponse};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::Path,
};
use wasmedge_stable_diffusion::{stable_diffusion_interface::ImageType, BaseFunction, Context};

/// Create an image given a prompt.
pub async fn image_generation(
    req: &mut ImageCreateRequest,
) -> Result<ListImagesResponse, LlamaCoreError> {
    let sd = match SD_TEXT_TO_IMAGE.get() {
        Some(sd) => sd,
        None => {
            let err_msg = "Fail to get the underlying value of `SD_TEXT_TO_IMAGE`.";

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let sd_locked = sd.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `SD_TEXT_TO_IMAGE`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match sd_locked.create_context().map_err(|e| {
        let err_msg = format!("Fail to create the context. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::InitContext(err_msg)
    })? {
        Context::TextToImage(mut text_to_image) => {
            // create a unique file id
            let id = format!("file_{}", uuid::Uuid::new_v4());

            // save the file
            let path = Path::new("archives");
            if !path.exists() {
                fs::create_dir(path).unwrap();
            }
            let file_path = path.join(&id);
            if !file_path.exists() {
                fs::create_dir(&file_path).unwrap();
            }
            let filename = "output.png";
            let output_image_file = file_path.join(filename);
            let output_image_file = output_image_file.to_str().unwrap();

            // create and dump the generated image
            text_to_image
                .set_prompt(&req.prompt)
                .set_output_path(output_image_file)
                .generate()
                .map_err(|e| {
                    let err_msg = format!("Fail to dump the image. {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // log
            #[cfg(feature = "logging")]
            info!(target: "llama_core", "file_id: {}, file_name: {}", &id, &filename);

            // convert the image to base64 string
            let base64_string = match image_to_base64(output_image_file) {
                Ok(base64_string) => base64_string,
                Err(e) => {
                    let err_msg = format!("Fail to convert the image to base64 string. {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };

            // log
            #[cfg(feature = "logging")]
            info!(target: "llama_core", "base64 string: {}", &base64_string.chars().take(10).collect::<String>());

            let created: u64 =
                match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        #[cfg(feature = "logging")]
                        error!(target: "llama_core", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.into()));
                    }
                };

            // create an image object
            let image = ImageObject {
                b64_json: Some(base64_string),
                url: None,
                prompt: Some(req.prompt.clone()),
            };

            Ok(ListImagesResponse {
                created,
                data: vec![image],
            })
        }
        _ => {
            let err_msg = "Fail to get the `TextToImage` context.";

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            Err(LlamaCoreError::Operation(err_msg.into()))
        }
    }
}

/// Create an edited or extended image given an original image and a prompt.
pub async fn image_edit(req: &mut ImageEditRequest) -> Result<ListImagesResponse, LlamaCoreError> {
    let sd = match SD_IMAGE_TO_IMAGE.get() {
        Some(sd) => sd,
        None => {
            let err_msg = "Fail to get the underlying value of `SD_IMAGE_TO_IMAGE`.";

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            return Err(LlamaCoreError::Operation(err_msg.into()));
        }
    };

    let sd_locked = sd.lock().map_err(|e| {
        let err_msg = format!("Fail to acquire the lock of `SD_IMAGE_TO_IMAGE`. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::Operation(err_msg)
    })?;

    match sd_locked.create_context().map_err(|e| {
        let err_msg = format!("Fail to create the context. {}", e);

        #[cfg(feature = "logging")]
        error!(target: "llama_core", "{}", &err_msg);

        LlamaCoreError::InitContext(err_msg)
    })? {
        Context::ImageToImage(mut image_to_image) => {
            // create a unique file id
            let id = format!("file_{}", uuid::Uuid::new_v4());

            // save the file
            let path = Path::new("archives");
            if !path.exists() {
                fs::create_dir(path).unwrap();
            }
            let file_path = path.join(&id);
            if !file_path.exists() {
                fs::create_dir(&file_path).unwrap();
            }
            let filename = "output.png";
            let output_image_file = file_path.join(filename);
            let output_image_file = output_image_file.to_str().unwrap();

            // get the path of the original image
            let origin_image_file = Path::new("archives")
                .join(&req.image.id)
                .join(&req.image.filename);
            let path_origin_image = origin_image_file.to_str().ok_or(LlamaCoreError::Operation(
                "Fail to get the path of the original image.".into(),
            ))?;

            // create and dump the generated image
            image_to_image
                .set_prompt(&req.prompt)
                .set_image(ImageType::Path(path_origin_image))
                .set_output_path(output_image_file)
                .generate()
                .map_err(|e| {
                    let err_msg = format!("Fail to dump the image. {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    LlamaCoreError::Operation(err_msg)
                })?;

            // log
            #[cfg(feature = "logging")]
            info!(target: "llama_core", "file_id: {}, file_name: {}", &id, &filename);

            // convert the image to base64 string
            let base64_string = match image_to_base64(output_image_file) {
                Ok(base64_string) => base64_string,
                Err(e) => {
                    let err_msg = format!("Fail to convert the image to base64 string. {}", e);

                    #[cfg(feature = "logging")]
                    error!(target: "llama_core", "{}", &err_msg);

                    return Err(LlamaCoreError::Operation(err_msg));
                }
            };

            // log
            #[cfg(feature = "logging")]
            info!(target: "llama_core", "base64 string: {}", &base64_string.chars().take(10).collect::<String>());

            let created: u64 =
                match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
                    Ok(n) => n.as_secs(),
                    Err(_) => {
                        let err_msg = "Failed to get the current time.";

                        // log
                        #[cfg(feature = "logging")]
                        error!(target: "llama_core", "{}", &err_msg);

                        return Err(LlamaCoreError::Operation(err_msg.into()));
                    }
                };

            // create an image object
            let image = ImageObject {
                b64_json: Some(base64_string),
                url: None,
                prompt: Some(req.prompt.clone()),
            };

            Ok(ListImagesResponse {
                created,
                data: vec![image],
            })
        }
        _ => {
            let err_msg = "Fail to get the `ImageToImage` context.";

            #[cfg(feature = "logging")]
            error!(target: "llama_core", "{}", &err_msg);

            Err(LlamaCoreError::Operation(err_msg.into()))
        }
    }
}

/// Create a variation of a given image.
pub async fn image_variation(
    _req: &mut ImageEditRequest,
) -> Result<ListImagesResponse, LlamaCoreError> {
    unimplemented!("image_variation")
}

// convert an image file to a base64 string
fn image_to_base64(image_path: &str) -> io::Result<String> {
    // Open the file
    let mut image_file = File::open(image_path)?;

    // Read the file into a byte array
    let mut buffer = Vec::new();
    image_file.read_to_end(&mut buffer)?;

    Ok(general_purpose::STANDARD.encode(&buffer))
}
