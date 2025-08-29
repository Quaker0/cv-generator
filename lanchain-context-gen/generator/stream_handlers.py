from langchain.callbacks.base import BaseCallbackHandler


class PrintAndCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.output = ""
        self.started = False

    def on_llm_start(self, *args, **kwargs):
        if not self.started:
            print("\n\033[96mThinking...\033[0m\n")  # Cyan header
            self.started = True

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
        self.output += token


class FileLoggingHandler(BaseCallbackHandler):
    def __init__(self, file_path):
        self.file_path = file_path
        self.buffer = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.buffer += token

    def on_llm_end(self, *args, **kwargs):
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(self.buffer)
