#note full codes about neural .json for query processing are under encoding-decoding processing before nbeing uploaded here to avoid infringements of ownership
#Remember to visit nobestudy.io for APIs access 
#Soon all .json will be uploaded and generative neural model to run it in your local PC under 8CPU

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class QAModelConfig:
    """Configuration for the QA model"""
    embedding_dim: int = 1024
    num_heads: int = 16
    spike_threshold: float = 0.5
    membrane_tau: float = 10.0
    synaptic_tau: float = 5.0
    max_sequence_length: int = 2048
    vocab_size: int = 50000

class SpikingAttention(nn.Module):
    """Spiking attention mechanism with temporal dynamics"""
    def __init__(self, config: QAModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.embedding_dim // config.num_heads
        
        # Spiking neuron parameters
        self.threshold = config.spike_threshold
        self.tau_mem = config.membrane_tau
        self.tau_syn = config.synaptic_tau
        
        # Neural projections
        self.query = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Membrane potentials
        self.register_buffer('membrane_potential', torch.zeros(1))
        self.register_buffer('synaptic_current', torch.zeros(1))

    def integrate_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """Temporal integration of input signals into spikes"""
        # Update synaptic current
        self.synaptic_current = self.synaptic_current * torch.exp(-1/self.tau_syn) + x
        
        # Update membrane potential
        self.membrane_potential = (self.membrane_potential * torch.exp(-1/self.tau_mem) + 
                                 self.synaptic_current)
        
        # Generate spikes
        spikes = (self.membrane_potential >= self.threshold).float()
        
        # Soft reset
        self.membrane_potential = self.membrane_potential - (spikes * self.threshold)
        
        return spikes

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        
        # Generate query, key, value projections
        q = self.query(x)
        k = self.key(context if context is not None else x)
        v = self.value(context if context is not None else x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.config.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.config.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.config.num_heads, self.head_dim)
        
        # Convert to spikes using temporal integration
        q_spikes = self.integrate_spikes(q)
        k_spikes = self.integrate_spikes(k)
        v_spikes = self.integrate_spikes(v)
        
        # Compute scaled dot-product attention with cosine similarity
        scale = math.sqrt(self.head_dim)
        q_normalized = F.normalize(q_spikes, dim=-1)
        k_normalized = F.normalize(k_spikes, dim=-1)
        
        # Compute cosine similarity-based attention scores
        attention_scores = torch.matmul(q_normalized, k_normalized.transpose(-2, -1)) / scale
        
        # Apply softmax and compute weighted values
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, v_spikes)
        
        # Reshape and project output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        output = context_layer.view(batch_size, seq_length, self.config.embedding_dim)
        
        return self.output(output)

class ZeroShotQABlock(nn.Module):
    """Main QA block combining spiking attention with zero-shot capabilities"""
    def __init__(self, config: QAModelConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.input_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.spiking_attention = SpikingAttention(config)
        
        # Knowledge integration layers
        self.knowledge_projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.context_integration = nn.Linear(config.embedding_dim * 2, config.embedding_dim)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)

    def encode_knowledge(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes external knowledge for zero-shot learning"""
        # Project knowledge into the same space as questions
        knowledge_encoded = self.knowledge_projection(x)
        
        # Apply spiking attention to knowledge
        knowledge_context = self.spiking_attention(knowledge_encoded)
        
        return self.layer_norm(knowledge_context)

    def forward(self, 
                question_ids: torch.Tensor,
                knowledge_base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for zero-shot question answering
        
        Args:
            question_ids: Tensor of token IDs for the question
            knowledge_base: Optional tensor containing relevant knowledge
            
        Returns:
            Tensor of logits for answer generation
        """
        # Get sequence info
        batch_size, seq_length = question_ids.shape
        device = question_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        
        # Compute embeddings
        token_embeddings = self.input_embedding(question_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings
        
        # Process knowledge if provided
        if knowledge_base is not None:
            knowledge_context = self.encode_knowledge(knowledge_base)
            # Integrate knowledge using spiking attention
            attention_output = self.spiking_attention(embeddings, knowledge_context)
        else:
            attention_output = self.spiking_attention(embeddings)
        
        # Final processing
        output = self.layer_norm(attention_output)
        logits = self.output_projection(output)
        
        return logits

    def answer_question(self, 
                       question: List[int],
                       knowledge: Optional[List[int]] = None,
                       max_length: int = 100) -> List[int]:
        """
        Generate an answer for a given question
        
        Args:
            question: List of token IDs for the question
            knowledge: Optional list of token IDs for knowledge context
            max_length: Maximum length of the generated answer
            
        Returns:
            List of token IDs for the generated answer
        """
        device = next(self.parameters()).device
        question_tensor = torch.tensor([question], device=device)
        
        if knowledge is not None:
            knowledge_tensor = torch.tensor([knowledge], device=device)
            knowledge_embeddings = self.input_embedding(knowledge_tensor)
        else:
            knowledge_embeddings = None
        
        # Generate answer tokens autoregressively
        generated = []
        for _ in range(max_length):
            logits = self.forward(question_tensor, knowledge_embeddings)
            next_token = torch.argmax(logits[0, -1]).item()
            generated.append(next_token)
            
            # Update input for next iteration
            question_tensor = torch.cat([
                question_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
            
            # Check for end of sequence
            if next_token == self.config.vocab_size - 1:  
                break
                
        return generated
