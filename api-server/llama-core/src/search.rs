use crate::error::SearchError;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Possible input/output Content Types. Currently only supports JSON.
#[derive(Debug, Eq, PartialEq)]
pub enum ContentType {
    JSON,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                ContentType::JSON => "application/json",
            }
        )
    }
}

/// The base Search Configuration holding all relevant information to access a search api and
/// retrieve results.
#[derive(Debug)]
pub struct SearchConfig {
    /// The search engine we're currently focusing on. Currently only one supported, to ensure
    /// stabiliity.
    #[allow(dead_code)]
    search_engine: String,
    /// The total number of results.
    max_search_results: u8,
    /// The size limit of every search result.
    size_limit_per_result: u16, // 128**2
    /// The endpoint for the search API.
    endpoint: String,
    /// The content type of the input.
    content_type: ContentType,
    /// The (expected) content type of the output.
    output_content_type: ContentType,
    /// Method expected by the api endpoint.
    method: String,
    //authentication: Option<AuthenticationMethod>,
    /// Additional headers for any other purpose.
    additional_headers: Option<std::collections::HashMap<String, String>>,
    /// Callback function to parse the output of the api-service. Implementation left to the user.
    parser: fn(&serde_json::Value) -> Result<SearchOutput, SearchError>,
}

impl SearchConfig {
    // wrapper for parse function
    pub fn parse_into_results(
        &self,
        raw_results: &serde_json::Value,
    ) -> Result<SearchOutput, SearchError> {
        (self.parser)(raw_results)
    }
    pub fn new(
        search_engine: String,
        max_search_results: u8,
        size_limit_per_result: u16,
        endpoint: String,
        content_type: ContentType,
        output_content_type: ContentType,
        method: String,
        additional_headers: Option<std::collections::HashMap<String, String>>,
        parser: fn(&serde_json::Value) -> Result<SearchOutput, SearchError>,
    ) -> SearchConfig {
        SearchConfig {
            search_engine,
            max_search_results,
            size_limit_per_result,
            endpoint,
            content_type,
            output_content_type,
            method,
            additional_headers,
            parser,
        }
    }
}

// output format for individual results in the final output.
#[derive(Serialize, Deserialize)]
pub struct SearchResult {
    pub url: String,
    pub site_name: String,
    pub text_content: String,
}

// Final output format for consumption by the LLM.
#[derive(Serialize, Deserialize)]
pub struct SearchOutput {
    pub results: Vec<SearchResult>,
}

impl SearchConfig {
    pub async fn perform_search<T: Serialize>(
        &self,
        search_input: &T,
    ) -> Result<SearchOutput, SearchError> {
        println!("entering SearchConfig");
        let client = Client::new();
        let url = match Url::parse(&self.endpoint) {
            Ok(url) => url,
            Err(_) => {
                let msg = "Malformed endpoind url";
                #[cfg(feature = "logging")]
                error!(target: "search", "perform_search: {}", msg);
                return Err(SearchError::Response(format!(
                    "When parsing endpoint: {}",
                    msg
                )));
            }
        };

        let method_as_string = match reqwest::Method::from_bytes(self.method.as_bytes()) {
            Ok(method) => method,
            _ => {
                let msg = "Header conversion failed";
                #[cfg(feature = "logging")]
                error!(target: "search", "perform_search: {}", msg);
                return Err(SearchError::Response(format!(
                    "When processing headers: {}",
                    msg
                )));
            }
        };

        let mut req = client.request(method_as_string, url);

        // check headers.
        req = req.headers(
            match (&self
                .additional_headers
                .clone()
                .unwrap_or_else(|| std::collections::HashMap::new()))
                .try_into()
            {
                Ok(headers) => headers,
                Err(_) => {
                    let msg = "Header parsing failed";
                    #[cfg(feature = "logging")]
                    error!(target: "search", "perform_search: {}", msg);
                    return Err(SearchError::Response(format!(
                        "When processing headers: {}",
                        msg
                    )));
                }
            },
        );

        req = match self.content_type {
            ContentType::JSON => req.json(search_input),
        };

        let res = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                #[cfg(feature = "logging")]
                error!(target: "search", "perform_search: {}", msg);
                return Err(SearchError::Response(format!(
                    "When recieving response: {}",
                    msg
                )));
            }
        };

        match res.content_length() {
            Some(length) => {
                if length == 0 {
                    let msg = "Empty repsonse from server";
                    #[cfg(feature = "logging")]
                    error!(target: "search", "perform_search: {}", msg);
                    return Err(SearchError::Response(format!(
                        "When recieving response: {}",
                        msg
                    )));
                }
            }
            None => {
                let msg = "Content length returned None";
                #[cfg(feature = "logging")]
                error!(target: "search", "perform_search: {}", msg);
                return Err(SearchError::Response(format!(
                    "When recieving response: {}",
                    msg
                )));
            }
        }

        // start parsing the output.
        let raw_results: Value;
        match self.output_content_type {
            ContentType::JSON => {
                let body_text = match res.text().await {
                    Ok(body) => body,
                    Err(e) => {
                        let msg = e.to_string();
                        #[cfg(feature = "logging")]
                        error!(target: "search", "perform_search: {}", msg);
                        return Err(SearchError::Response(format!(
                            "When accessing response body: {}",
                            msg
                        )));
                    }
                };
                println!("{}", body_text);
                raw_results = match serde_json::from_str(body_text.as_str()) {
                    Ok(value) => value,
                    Err(e) => {
                        let msg = e.to_string();
                        #[cfg(feature = "logging")]
                        error!(target: "search", "perform_search: {}", msg);
                        return Err(SearchError::Response(format!(
                            "When converting to a JSON object: {}",
                            msg
                        )));
                    }
                };
            }
        };

        // start cleaning the output.

        // produce SearchOutput instance with the raw results obtained from the endpoint.
        let mut search_output: SearchOutput = match self.parse_into_results(&raw_results) {
            Ok(search_output) => search_output,
            Err(e) => {
                let msg = e.to_string();
                #[cfg(feature = "logging")]
                error!(target: "search", "perform_search: {}", msg);
                return Err(SearchError::Response(format!(
                    "When accessing response body: {}",
                    msg
                )));
            }
        };

        // apply maximum search result limit.
        search_output
            .results
            .truncate(self.max_search_results as usize);

        // apply per result character limit.
        for result in search_output.results.iter_mut() {
            if let Some(clipped_content) = result
                .text_content
                .split_at_checked(self.size_limit_per_result as usize)
            {
                result.text_content = clipped_content.0.to_string();
            }
        }

        // Search Output cleaned and finalized.
        Ok(search_output)
    }
}
