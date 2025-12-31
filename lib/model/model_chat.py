import ollama

class ModelChat():
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        print(f"Chat with {self.model_name} model is started !!!")

    def generate_zero_shot_answer(self, user_content, system_instruction=None):

        messages = []

        # 1. Sistem Rolü (Varsa ekle)
        if system_instruction:
            messages.append({
                'role': 'system',
                'content': system_instruction
            })

        # 2. Kullanıcı Rolü
        messages.append({
            'role': 'user',
            'content': user_content
        })

        try:
            response = ollama.chat(model=self.model_path, messages=messages)
            return response['message']['content']
        except Exception as e:
            print(f"Model hatası: {e}")
            return ""
