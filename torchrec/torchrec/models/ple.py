from torch import nn
from torch.nn import functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class Gate(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

class PLE(nn.Module):
    def __init__(self, input_dim, shared_experts, task_experts, task_dims):
        super(PLE, self).__init__()
        
        # Shared experts
        self.shared_experts = nn.ModuleList([Expert(input_dim, task_dims[0]) for _ in range(shared_experts)])
        
        # Task-specific experts
        self.task_experts = nn.ModuleList([nn.ModuleList([Expert(input_dim, task_dims[0]) for _ in range(task_experts)]) for _ in task_dims])
        
        # Gating networks for each task
        self.gates = nn.ModuleList([Gate(input_dim, shared_experts + task_experts) for _ in task_dims])
        
        # Task-specific towers
        self.task_towers = nn.ModuleList([nn.Linear(task_dims[0], task_dim) for task_dim in task_dims])

    def forward(self, x):
        # Shared expert outputs
        shared_outputs = [expert(x) for expert in self.shared_experts]
        
        # Task-specific expert outputs
        task_outputs = [[expert(x) for expert in experts] for experts in self.task_experts]

        final_outputs = []
        for i, gate in enumerate(self.gates):
            gate_outputs = gate(x)
            expert_outputs = shared_outputs + task_outputs[i]
            
            # Combine outputs from shared and task-specific experts based on gate outputs
            output = sum(gate_outputs[:, j].unsqueeze(1) * expert_outputs[j] for j in range(len(expert_outputs)))
            
            # Process through task-specific tower
            final_outputs.append(self.task_towers[i](output))

        return final_outputs