# GPT-Neo Flask Demo

A simple flask website for demoing GPT-Neo.

## Build and run

```bash
docker build -t gpt-neo-flask-demo .
docker run \
    --name gpt-neo-flask-demo \
    -p 5000:5000 \
    --restart on-failure \
    gpt-neo-flask-demo
```