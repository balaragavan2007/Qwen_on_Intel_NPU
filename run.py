import openvino_genai
from openvino_genai import LLMPipeline

class ChatModel:
    """
    A simplified, non-streaming chatbot focused on getting a complete response.
    """
    def __init__(self, model_path: str, device: str = "AUTO"):
        print(f"Loading model on device: {device}...")

        pipeline_args = {
            'models_path': model_path,
            'device': device,
            'collect_perf_metrics': True
        }
        print("Performance metrics collection is enabled.")

        self.pipe = LLMPipeline(**pipeline_args)
        
        self.config = openvino_genai.GenerationConfig()
        self.config.max_new_tokens = 1024
        
        self.history = [] # We bring back history for a better conversation
        
        print("Model loaded and ready.")
        
    def _format_performance_metrics(self, perf_metrics_obj) -> dict:
        """Helper method to format the performance metrics object."""
        if not perf_metrics_obj:
            return None
            
        return {
            'ttft (s)': round(perf_metrics_obj.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics_obj.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics_obj.get_throughput().mean, 2),
            'new_tokens': perf_metrics_obj.get_num_generated_tokens(),
        }

    def generate(self, prompt: str) -> dict:
        """
        Generates a full response in a single, blocking call.
        """
        self.history.append(prompt)
        
        # We make a simple, non-streaming call to generate.
        result_obj = self.pipe.generate(self.history, self.config)
        
        response_text = result_obj.texts[0]
        performance_metrics = self._format_performance_metrics(result_obj.perf_metrics)
        
        if '</think>' in response_text:
            response_text = response_text.split('</think>')[-1].strip()
        
        self.history.append(response_text)

        return {
            "text": response_text,
            "performance": performance_metrics
        }

# --- Main Application Logic ---
if __name__ == "__main__":
    MODEL_PATH = "Qwen3-8B-NPU-Model"
    
    model = ChatModel(model_path=MODEL_PATH, device="NPU")
    
    print("\n--- Chatbot Ready ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response_data = model.generate(user_input)
        bot_text = response_data["text"]
        perf_metrics = response_data["performance"]

        # The bot will "think" for a moment, then print the full answer at once.
        print(f"Bot: {bot_text}")
        
        if perf_metrics:
            print("\n--- Performance ---")
            key_width = max(len(key) for key in perf_metrics.keys())
            for key, value in perf_metrics.items():
                print(f"  {key.ljust(key_width)} : {value}")
            print("-------------------")

    print("\nChat session ended.")