import torch

class CustomLossFunction:
    def __init__(self, reward_smoothing_factor: float = 0.9):
        """
        Args:
            reward_smoothing_factor: EMA için α değeri. 0.0 yakınsa
                sadece en son batch ödülleri baz alınır; 1.0 yakınsa
                taban çizgisi çok yavaş güncellenir.
        """
        self.running_reward_baseline = 0.0
        self.reward_smoothing_factor = reward_smoothing_factor

    def __call__(self,
                 log_probabilities: torch.Tensor,
                 rewards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            log_probabilities: Tensor(shape=[batch_size]) — her örnek
                               için eylem log-olasılıkları (toplamı).
            rewards:           Tensor(shape=[batch_size]) — her örnek
                               için elde edilen ödül.
        Returns:
            torch.Tensor: tek skaler RL loss (ortalama).
        """

        if not isinstance(log_probabilities, torch.Tensor):
            log_probabilities = torch.tensor(log_probabilities)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards)

        log_probabilities = log_probabilities.detach().requires_grad_(True)
        rewards = rewards.detach().requires_grad_(True)


        # 1) Batch ödül ortalamasını al
        batch_mean_reward = rewards.mean().item()

        # 2) EMA taban çizgisini güncelle
        self.running_reward_baseline = (
            self.reward_smoothing_factor * self.running_reward_baseline
            + (1.0 - self.reward_smoothing_factor) * batch_mean_reward
        )

        # 3) Avantaj vektörünü hesapla: r_i - baseline
        #    (broadcast olur)
        advantages = rewards - self.running_reward_baseline
        #
        # print(advantages)
        #
        # print(log_probabilities)
        # 4) Policy-gradient kaybı: -log π(a|s) * A
        #    örnek başına, sonra ortalama alıyoruz
        per_sample_loss = - log_probabilities * advantages

        # 5) İstersen toplam yerine ortalama dönebilirsin
        loss = per_sample_loss.mean()

        return loss
