# GPT-Neo Flask Demo

A simple flask website for demoing GPT-Neo.

## Build and run

```bash
docker build -t gpt-neo-flask-demo .
docker run -d \
    --name gpt-neo-flask-demo \
    -p 5000:5000 \
    --restart on-failure \
    -v /path/for/cache:/.cache/huggingface/transformers \
    gpt-neo-flask-demo
```