# ----------------------------------------------------------------------------
# This file contains the SelectorGYM class, which is a class that is used to train selector models.
# ----------------------------------------------------------------------------

from src.selectors.GCNSelector import GCNSelector
from src.selectors.GATSelector import GATSelector
from src.prompt_factories.PromptFactory import PromptFactory

from typing import List, Tuple, Optional
import time

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data

import google.generativeai as genai


class SelectorGYM():
    def __init__(self, selector_type: str,
                 n_select: int,
                 genai_model_name: str,
                 prompt_factory: PromptFactory,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame):

        # Initialize the dataframes
        self.df_train = df_train
        self.df_test = df_test

        # Initialize the selector model
        if selector_type == "GCN":
            self.selector = GCNSelector(in_channels=768, hidden_channels=128)
        elif selector_type == "GAT":
            self.selector = GATSelector(in_channels=768, hidden_channels=128)
        else:
            raise ValueError("Unsupported selector type")

        # Number of nodes to select
        self.n_select = n_select

        self.selector_path = f"/Users/cagatay/Desktop/CS/Projects/BioLaySumm-BiOzU/models/GCN_20_selector.pth"

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.selector.parameters(), lr=0.01)

        # Initialize the Generative AI model
        self.genai_model_name = genai_model_name
        self.genai_model = genai.GenerativeModel(self.genai_model_name)

        # Initialize the prompt factory
        self.prompt_factory = prompt_factory

        # Global baseline variables for Exponential Moving Average (EMA)
        self.running_reward_baseline = 0.0  # Initial baseline value
        self.reward_smoothing_factor = 0.99  # Decay rate for EMA

        # Error counters for debugging
        self.train_error_count = 0
        self.test_error_count = 0

        # Train loss history for analysis train performance
        self.train_loss_history = []

    def reset(self):
        self.test_error_count = 0
        self.train_error_count = 0
        self.reset_generaoi_model()

    def reset_generaoi_model(self):
        """
        Reset the Generative AI model with a new model name.
        """
        print("Resetting Generative AI model...")
        self.genai_model = genai.GenerativeModel(self.genai_model_name)

    def get_graphs(self, row: pd.Series) -> Tuple[Data, List[str]]:
        """
        Generate a graph representation from the given row of dataframe

        Args:
            row (pd.Series): A row of dataframe that contains the sections and their embeddings

        Returns:
            Tuple[Data, List[str]]: A tuple that contains the graph data and the sentences of the sections
        """

        # Flaten the nodes and sentences
        nodes = [x for y in row['sections_embedding'] for x in y]
        sentences = [x for y in row['sections'] for x in y]

        # Calculate all cosine similarity values
        similarities = cosine_similarity(nodes)

        # Calculate the average cosine similarity as a threshold
        avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

        # Select the edges that have similarity greater than the average similarity
        edges = [
            (i, j, similarities[i, j])
            for i in range(len(nodes)) for j in range(len(nodes))
            if i != j and similarities[i, j] > avg_similarity
        ]

        # Create the edge index tensor
        edges_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()

        # Create the graph data
        data = Data(x=torch.tensor(nodes, dtype=torch.float), edge_index=edges_index)

        return data, sentences

    def calculate_rouge_reward(self, predicted_text: str, reference_text: str) -> float:
        """
        Computes the ROUGE-based reward for reinforcement learning.

        Args:
            predicted_text (str): The generated summary text.
            reference_text (str): The ground-truth summary text.

        Returns:
            float: The average ROUGE-1 and ROUGE-L F1 score.
        """
        # Initialize the ROUGE scorer with stemming enabled
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        # Compute ROUGE scores between reference and predicted summaries
        rouge_scores = rouge_scorer_obj.score(reference_text, predicted_text)

        # Compute the average F1 score for ROUGE-1 and ROUGE-L as the final reward
        average_rouge_f1 = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2.0

        return average_rouge_f1

    def compute_rl_loss(self, log_probabilities: torch.Tensor,
                        generated_summary: str,
                        ground_truth_summary: str) -> torch.Tensor:
        """
        Computes the REINFORCE-style loss function using an Exponential Moving Average (EMA) baseline.

        This function applies policy gradient loss to train the model, adjusting for variance
        using a running baseline.

        Args:
            log_probabilities (torch.Tensor): The sum of log probabilities of the selected actions.
            generated_summary (str): The summary generated by the model.
            ground_truth_summary (str): The reference summary (ground truth).

        Returns:
            torch.Tensor: The computed reinforcement learning loss.
        """

        # Compute ROUGE reward based on the similarity of generated and reference summaries
        reward_score = self.calculate_rouge_reward(generated_summary, ground_truth_summary)

        # Update the baseline using Exponential Moving Average (EMA)
        self.running_reward_baseline = (
                self.reward_smoothing_factor * self.running_reward_baseline +
                (1 - self.reward_smoothing_factor) * reward_score
        )

        # Compute the advantage by subtracting the running baseline from the current reward
        advantage = reward_score - self.running_reward_baseline

        # Compute policy gradient loss (REINFORCE algorithm)
        rl_loss = -log_probabilities * advantage

        return rl_loss

    def generate_summary(self, row: pd.Series, max_retries: int = 3, retry_delay: float = 5.0) -> str:
        """
        Generates a lay summary using the Gemini AI model.
        Includes automatic retries in case of API failures.

        Args:
            row (pd.Series): A row of dataframe containing the data needed for prompt generation.
            max_retries (int): Maximum number of retries for handling API errors.
            retry_delay (float): Time (in seconds) to wait before retrying after a failure.

        Returns:
            Optional[str]: The generated summary if successful, None if the request fails.
        """
        for attempt in range(1, max_retries + 1):
            try:
                # Generate prompt from input data
                self.prompt_factory.set_row(row)
                prompt_text = self.prompt_factory.get_prompt()

                # Request AI model to generate content
                response = self.genai_model.generate_content(prompt_text)

                # Extract and return summary from response
                return self.extract_summary(response.text)

            except Exception as error:
                print(f"Attempt {attempt} failed: {error}")

                # If it's the last attempt, return None
                if attempt == max_retries:
                    print("Maximum retries reached. Returning None.")
                    return ""

                if attempt % 2 == 0:
                    # Initialize the Generative AI model
                    self.reset_generaoi_model()

                # Wait before retrying the request
                time.sleep(retry_delay)

    def extract_summary(self, response_text: str) -> str:
        """
        Extracts the summary from the AI model response.

        Args:
            response_text (str): The raw response text from the AI model.

        Returns:
            str: Extracted summary text.
        """
        # Check for different possible summary formats in the response
        if "lay_summary:" in response_text:
            return response_text.split("lay_summary:")[1].strip()
        elif "**Lay Summary:**" in response_text:
            return response_text.split("**Lay Summary:**")[1].strip()
        else:
            return response_text.strip()

    def train_step(self, row: pd.Series) -> torch.Tensor:
        """
        Perform a single training step for the selector model using the REINFORCE algorithm.

        Args:
            row (pd.Series): A row of dataframe containing the data needed for training.

        Returns:
            torch.Tensor: Loss function results.
        """

        self.selector.train()
        self.optimizer.zero_grad()

        data, sentences = self.get_graphs(row)
        reference_summary = row['summary']

        # Forward pass: get logits per node
        logits = self.selector(data.x, data.edge_index)  # shape: [num_nodes]

        # Convert logits to probabilities with softmax.
        probs = F.softmax(logits, dim=0)  # shape: [num_nodes]

        # Sample n_select unique nodes based on the probability distribution.
        node_indices = torch.topk(probs, self.n_select).indices

        # Sum the log probabilities of the selected nodes.
        # (This is our “policy log–probability”.)
        selected_log_probs = torch.log(probs[node_indices]).sum()

        # Build the prompt by concatenating the sentences for the selected nodes.
        selected_sentences = [sentences[i] for i in node_indices.tolist()]
        row['rag_sentences'] = selected_sentences

        # Generate the summary using the selected sentences.
        generated_summary = self.generate_summary(row)

        # Compute the loss using the REINFORCE algorithm if a summary is generated successfully.
        if not generated_summary == "":
            loss = self.compute_rl_loss(selected_log_probs, generated_summary, reference_summary)

            loss.backward()
            self.optimizer.step()
        else:
            self.train_error_count += 1
            loss = torch.tensor(0.0)
        return loss

    def test_step(self, row: pd.Series) -> Tuple[torch.Tensor, str, str]:
        """
        Perform a single testing step for the selector model using the REINFORCE algorithm.

        Args:
            row (pd.Series): A row of dataframe containing the data needed for testing.

        Returns:
            Tuple[torch.Tensor, str]: Loss function results and the generated summary.
        """

        self.selector.eval()

        with torch.no_grad():
            data, sentences = self.get_graphs(row)
            reference_summary = row['summary']

            # Forward pass: get logits per node
            logits = self.selector(data.x, data.edge_index)

            # Convert logits to probabilities with softmax.
            probs = F.softmax(logits, dim=0)

            # Sample n_select unique nodes based on the probability distribution.
            node_indices = torch.topk(probs, self.n_select).indices

            # Sum the log probabilities of the selected nodes.
            # (This is our “policy log–probability”.)
            selected_log_probs = torch.log(probs[node_indices]).sum()

            # Build the prompt by concatenating the sentences for the selected nodes.
            selected_sentences = [sentences[i] for i in node_indices.tolist()]
            row['rag_sentences'] = selected_sentences

            # Generate the summary using the selected sentences.
            generated_summary = self.generate_summary(row)

            # Compute the loss using the REINFORCE algorithm if a summary is generated successfully.
            if not generated_summary == "":
                loss = self.compute_rl_loss(selected_log_probs, generated_summary, reference_summary)
            else:
                self.test_error_count += 1
                loss = torch.tensor(0.0)

        return loss, generated_summary, reference_summary

    def train(self, n_epochs: int):
        """
        Train the selector model using the REINFORCE algorithm.

        Args:
            n_epochs (int): Number of training epochs.
        """

        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            self.reset()
            train_loss = 0.0
            num_success = 0

            for idx, row in self.df_train.iterrows():
                loss = self.train_step(row)
                train_loss += loss
                if loss > 0:
                    num_success += 1
                if idx % 2 == 0 or idx == len(self.df_train) - 1:
                    print(
                        f"\rEpoch {epoch + 1}/{n_epochs} | Process {idx + 1}/{len(self.df_train)} - {round(((idx + 1) / len(self.df_train) * 100), 2)}% | Loss: {loss:.4f} | Error Count: {self.train_error_count}", end=" "
                        )
            avg_train_loss = train_loss / num_success
            self.train_loss_history.append(avg_train_loss)
            print(f"\nEpoch {epoch + 1} Completed| Avg Train Loss: {avg_train_loss}")

        print("Training completed.")

    def test(self):
        """
        Test the selector model using the REINFORCE algorithm.
        """
        test_loss = 0.0
        predicted_summaries = []
        referance_summaries = []

        for idx, row in self.df_test.iterrows():
            loss, predicted_summary, referance_summary = self.test_step(row)
            test_loss += loss
            predicted_summaries.append(predicted_summary)
            referance_summaries.append(referance_summary)
            print(
                f"Process {idx + 1}/{len(self.df_test)} - {round(((idx + 1) / len(self.df_test) * 100), 2)}% | Test Loss: {loss} | Error Count: {self.test_error_count}",
                )

        print(f"Test Loss: {test_loss}")
        return predicted_summaries, referance_summaries

    def save_selector(self):
        """
        Save the trained selector model.
        """
        torch.save(self.selector.state_dict(), self.selector_path)
        print("Selector model saved.")

    def load_selector(self):
        """
        Load the trained selector model.
        """
        self.selector.load_state_dict(torch.load(self.selector_path, weights_only=True))
        print("Selector model loaded.")

    def plot_training_loss(self, loss_history):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Average Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.grid()
        plt.show()

    def get_train_loss_history(self) -> List[float]:
        """
        Get the training loss history.

        Returns:
            List[float]: List of training loss values.
        """
        return self.train_loss_history

    def get_path_selector(self):
        return self.selector_path

    def set_path_selector(self, path: str):
        self.selector_path = path
