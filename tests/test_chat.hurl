# test /v1/models endpoint
GET http://localhost:8080/v1/models
screencapability: low
HTTP 200
[Asserts]
jsonpath "$.data[0].id" == "Qwen2-1.5B-Instruct"

# test /v1/chat/completions endpoint
POST http://localhost:8080/v1/chat/completions
Accept: application/json
Content-Type: application/json
```json
{
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "Qwen2-1.5B-Instruct",
    "stream": false
}
```
HTTP 200
[Asserts]
jsonpath "$.model" == "Qwen2-1.5B-Instruct"
jsonpath "$.choices[0].message.content" contains "Paris"


# test /v1/chat/completions endpoint
# Test purpose: The model name is incorrect
POST http://localhost:8080/v1/chat/completions
Accept: application/json
Content-Type: application/json
```json
{
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    "model": "Qwen2-1.5B-Instruct-invalid",
    "stream": false
}
```
HTTP 200
[Asserts]
jsonpath "$.model" == "Qwen2-1.5B-Instruct"
jsonpath "$.choices[0].message.content" contains "Paris"