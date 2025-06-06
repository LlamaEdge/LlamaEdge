//! `endpoints` is part of [LlamaEdge API Server](https://github.com/LlamaEdge/LlamaEdge/tree/main/api-server) project. It defines the data types which are derived from the [OpenAI API Reference](https://platform.openai.com/docs/api-reference).

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[cfg(feature = "logging")]
#[macro_use]
extern crate log;

pub mod audio;
pub mod chat;
pub mod common;
pub mod completions;
pub mod embeddings;
pub mod files;
pub mod images;
pub mod models;
#[cfg(any(feature = "rag", feature = "index"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "rag", feature = "index"))))]
pub mod rag;
