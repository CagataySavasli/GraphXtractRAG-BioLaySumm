class AlternatingTraining:
    def __init__(self, selector_gym, gemini_gym, num_episodes=10):
        self.selector_gym = selector_gym
        self.gemini_gym = gemini_gym

    def alternating_training(self, num_episodes=10):
        """
        Alternating training between SelectorGYM and GeminiGYM for num_episodes.

        Args:
            num_episodes (int): Number of epochs for alternating training.
        """
        iter_epoch = 1
        for epoch in range(num_episodes):
            print(f"Epoch {epoch + 1}/{num_episodes} | Iteration {iter_epoch}")
            if epoch % 2 == 0:
                print("Fine-tuning GeminiGYM...")
                self.gemini_gym.fine_tune(
                    display_name=f"graphxtractrag_{epoch}",
                    epoch_count=iter_epoch
                )
                self.gemini_gym.update_model()

            else:
                print("Training SelectorGYM...")
                self.selector_gym.train(iter_epoch)  # Train selector for 1 epoch
                self.selector_gym.save_selector()
                iter_epoch += 1

            self.update_models(epoch)

        self.gemini_gym.fine_tune(
            display_name=f"graphxtractrag_final",
            epoch_count=5
        )
        self.gemini_gym.update_model()
    def update_models(self, epoch):
        print("#" * 50)
        print(self.gemini_gym.base_model)
        print(self.selector_gym.genai_model_name)
        print("-" * 50)
        self.selector_gym.genai_model_name = self.gemini_gym.base_model
        self.selector_gym.reset_generaoi_model()

        print(self.gemini_gym.base_model)
        print(self.selector_gym.genai_model_name)
        print(f"Epoch {epoch + 1} completed.\n")
