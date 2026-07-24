"""
Expert Discussion Implementation for Method Design
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class DiscussionMessage:
    expert_name: str
    content: str
    message_type: str  # 'proposal', 'feedback', 'question', 'answer', 'critique'
    round: int
    references: List[str] = None  # References to other experts' messages
    confidence_score: float = 0.0

class ExpertDiscussion:
    def __init__(self, llm_client, rag_retriever=None):
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.discussion_history = []
        self.current_round = 0
        self.knowledge_cache = {}
        self.fast_mode = os.getenv("METHOD_DESIGN_FAST_MODE", "true").lower() in {"1", "true", "yes"}
        self.max_tokens_per_call = int(os.getenv("METHOD_DESIGN_MAX_TOKENS_PER_CALL", "350"))
        self.token_budget = int(os.getenv("METHOD_DESIGN_TOKEN_BUDGET", "120000"))
        self.token_stats = {"calls": 0, "prompt_tokens_est": 0, "completion_tokens_est": 0}

        if not hasattr(ExpertDiscussion, '_llm_validated'):
            self._validate_llm_config()
            ExpertDiscussion._llm_validated = True

    def generate_expert_prompt(self, expert_name: str, expert_domain: str,
                             task_analysis: Dict[str, Any],
                             message_type: str,
                             context: List[DiscussionMessage] = None) -> str:
        """Generate context-aware prompt for expert discussion"""

        required_fields = [
            'task_type', 'dataset', 'perturbations', 'cell_types',
            'objectives', 'constraints', 'evaluation_metrics'
        ]
        missing_fields = [f for f in required_fields if f not in task_analysis]
        if missing_fields:
            print(f"Warning: Task analysis missing fields: {', '.join(missing_fields)}")

            task_analysis = self._fill_missing_fields(task_analysis, missing_fields)

        task_description = f"""Task Type: {task_analysis['task_type']}
Dataset: {task_analysis['dataset']['name']} ({task_analysis['dataset'].get('type', 'Unknown type')})
Dataset Description: {task_analysis['dataset'].get('description', 'No description provided')}

Perturbations:
{self._format_perturbations(task_analysis['perturbations'])}

Cell Types: {', '.join(task_analysis['cell_types'])}

Objectives:
{self._format_list(task_analysis['objectives'])}

Constraints:
{self._format_list(task_analysis['constraints'])}

Evaluation Metrics:
{self._format_list(task_analysis['evaluation_metrics'])}"""

        knowledge_content = self._get_relevant_knowledge(
            expert_domain=expert_domain,
            task_type=task_analysis['task_type'],
            message_type=message_type
        )

        # Base prompt with expert role and detailed task
        base_prompt = f"""You are {expert_name}, an expert in {expert_domain}.

Your task is to design a state-of-the-art model architecture for single cell perturbation prediction that outperforms current baselines. Your expertise in {expert_domain} is crucial for this task.

TASK DETAILS:
{task_description}

DESIGN GOALS:
1. The architecture must be specifically optimized for {task_analysis['task_type']} in single-cell data
2. The design should incorporate domain-specific knowledge about cellular responses
3. The model should achieve better performance than current SOTA baselines
4. The architecture should be interpretable and biologically meaningful
5. The implementation should be computationally efficient and scalable

Your role is to contribute your expertise in {expert_domain} to ensure these goals are met."""

        if knowledge_content:
            base_prompt += f"\n\nRELEVANT KNOWLEDGE:\n{knowledge_content}"

        # Add context from previous messages if available
        if context:
            context_str = "\nPrevious Discussion:\n"
            for msg in context[-5:]:  # Last 5 messages for context
                context_str += f"{msg.expert_name}: {msg.content}\n"
            base_prompt += context_str

        domain_context = self._get_domain_context(expert_domain)
        base_prompt += f"\n\nDOMAIN CONTEXT:\n{domain_context}"

        if message_type == 'proposal':
            base_prompt += """
Please propose a detailed architectural solution focusing on your domain expertise. Your proposal should aim to surpass current SOTA models in single-cell perturbation prediction.

Required sections:

1. Architecture Components:
   - Detailed description of each component
   - Configuration parameters and their justification
   - How each component addresses specific challenges in single-cell data
   - Integration points with other components

2. Biological Considerations:
   - How the architecture handles cell type specificity
   - Incorporation of pathway knowledge
   - Mechanisms for capturing perturbation effects
   - Biological constraints and their implementation

3. Technical Implementation:
   - Detailed layer specifications
   - Activation functions and their biological relevance
   - Loss function design
   - Training strategy recommendations

4. Performance Optimization:
   - Specific techniques to outperform SOTA
   - Computational efficiency considerations
   - Scalability solutions
   - Memory optimization strategies

5. Integration and Interpretability:
   - Methods for model interpretation
   - Visualization approaches
   - Integration with existing biological knowledge
   - Validation strategies

Format your response as a structured proposal with clear sections and specific technical details."""

        elif message_type == 'feedback':
            base_prompt += """
Please provide detailed feedback on the previous proposals, focusing on improving model performance beyond SOTA:

1. Technical Analysis:
   - Architectural design evaluation
   - Component integration assessment
   - Performance bottleneck identification
   - Scalability analysis
   - Memory efficiency review

2. Biological Relevance:
   - Cell type specificity assessment
   - Pathway integration evaluation
   - Perturbation response mechanism analysis
   - Biological constraint compliance
   - Interpretability assessment

3. Implementation Feasibility:
   - Technical complexity evaluation
   - Resource requirements analysis
   - Development timeline estimation
   - Risk assessment
   - Mitigation strategies

4. Improvement Suggestions:
   - Specific architectural enhancements
   - Performance optimization techniques
   - Integration improvements
   - Biological accuracy improvements
   - Implementation recommendations

5. Comparative Analysis:
   - Comparison with SOTA approaches
   - Identification of competitive advantages
   - Performance gap analysis
   - Innovation assessment
   - Potential impact evaluation

Be specific, technical, and constructive in your feedback. Focus on actionable improvements that can help surpass SOTA performance."""

        elif message_type == 'question':
            base_prompt += """
Based on your domain expertise, ask critical questions about the current proposals:

1. Technical Clarifications:
   - Architecture design choices
   - Implementation details
   - Performance optimization strategies
   - Integration mechanisms
   - Scaling approaches

2. Biological Considerations:
   - Cell type handling
   - Pathway integration methods
   - Perturbation response modeling
   - Biological constraint implementation
   - Validation approaches

3. Performance Questions:
   - SOTA comparison methods
   - Efficiency improvements
   - Accuracy enhancements
   - Scalability solutions
   - Resource optimization

4. Integration Concerns:
   - Component interactions
   - Data flow
   - Information sharing
   - Cross-module dependencies
   - System coherence

5. Validation and Testing:
   - Performance metrics
   - Biological validation
   - Technical verification
   - Quality assurance
   - Benchmark comparisons

Frame your questions to drive the discussion toward concrete improvements and SOTA-surpassing solutions."""

        elif message_type == 'critique':
            base_prompt += """
As a scientific critic, provide a comprehensive evaluation of the current proposals:

1. Scientific Merit (Score 0.1-1.0):
   - Theoretical foundation
   - Methodological rigor
   - Innovation level
   - Technical sophistication
   - Biological relevance

2. Technical Feasibility (Score 0.1-1.0):
   - Implementation complexity
   - Resource requirements
   - Scalability potential
   - Performance expectations
   - Maintenance considerations

3. Biological Validity (Score 0.1-1.0):
   - Cell type handling
   - Pathway integration
   - Perturbation modeling
   - Constraint compliance
   - Interpretability

4. SOTA Comparison (Score 0.1-1.0):
   - Performance potential
   - Innovation aspects
   - Technical advantages
   - Biological accuracy
   - Overall competitiveness

5. Implementation Strategy (Score 0.1-1.0):
   - Development approach
   - Resource allocation
   - Timeline realism
   - Risk management
   - Quality assurance

Provide a detailed justification for each score and an overall score (0.1-1.0) weighted across all categories.
Focus on potential for surpassing SOTA performance in single-cell perturbation prediction."""

        return base_prompt

    def generate_expert_message(self, expert_name: str, expert_domain: str,
                              task_analysis: Dict[str, Any],
                              message_type: str,
                              context: List[DiscussionMessage] = None) -> DiscussionMessage:
        """Generate a message from an expert using LLM"""

        # Generate appropriate prompt
        prompt = self.generate_expert_prompt(
            expert_name=expert_name,
            expert_domain=expert_domain,
            task_analysis=task_analysis,
            message_type=message_type,
            context=context
        )

        try:
            # Get response from LLM
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=self.max_tokens_per_call
            )

            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(response) // 4)
            self.token_stats["calls"] += 1
            self.token_stats["prompt_tokens_est"] += prompt_tokens
            self.token_stats["completion_tokens_est"] += completion_tokens
            total_est = self.token_stats["prompt_tokens_est"] + self.token_stats["completion_tokens_est"]
            print(
                f"[MethodDesign] LLM call={self.token_stats['calls']} "
                f"prompt~{prompt_tokens} completion~{completion_tokens} total~{total_est}"
            )
            if total_est > self.token_budget:
                raise RuntimeError(
                    f"Token budget exceeded (~{total_est}>{self.token_budget}). "
                    "Increase METHOD_DESIGN_TOKEN_BUDGET or reduce rounds/experts."
                )

            # Create discussion message
            message = DiscussionMessage(
                expert_name=expert_name,
                content=response,
                message_type=message_type,
                round=self.current_round,
                references=[msg.expert_name for msg in (context or [])]
            )

            # Add to discussion history
            self.discussion_history.append(message)

            return message

        except Exception as e:
            print(f"Error generating expert message: {e}")
            # Return a fallback message
            return DiscussionMessage(
                expert_name=expert_name,
                content=f"Error generating response: {str(e)}",
                message_type=message_type,
                round=self.current_round
            )

    def run_discussion_round(self, experts: List[Dict[str, str]],
                           task_analysis: Dict[str, Any]) -> List[DiscussionMessage]:
        """Run a complete discussion round with all experts"""
        round_messages = []
        if self.fast_mode and len(experts) > 2:
            experts = experts[:2]

        # 1. Initial proposals
        for expert in experts:
            proposal = self.generate_expert_message(
                expert_name=expert['name'],
                expert_domain=expert['domain'],
                task_analysis=task_analysis,
                message_type='proposal',
                context=self.discussion_history[-5:] if self.discussion_history else None
            )
            round_messages.append(proposal)

        # 2. Expert feedback
        for expert in experts:
            feedback = self.generate_expert_message(
                expert_name=expert['name'],
                expert_domain=expert['domain'],
                task_analysis=task_analysis,
                message_type='feedback',
                context=round_messages
            )
            round_messages.append(feedback)

        # 3. Questions and clarifications
        if self.fast_mode:
            if experts:
                questioner = experts[0]
                question = self.generate_expert_message(
                    expert_name=questioner['name'],
                    expert_domain=questioner['domain'],
                    task_analysis=task_analysis,
                    message_type='question',
                    context=round_messages
                )
                round_messages.append(question)
            if len(experts) > 1:
                responder = experts[1]
                answer = self.generate_expert_message(
                    expert_name=responder['name'],
                    expert_domain=responder['domain'],
                    task_analysis=task_analysis,
                    message_type='answer',
                    context=round_messages[-5:]
                )
                round_messages.append(answer)
        else:
            for expert in experts:
                question = self.generate_expert_message(
                    expert_name=expert['name'],
                    expert_domain=expert['domain'],
                    task_analysis=task_analysis,
                    message_type='question',
                    context=round_messages
                )
                round_messages.append(question)

                # Get answers from other experts
                for other_expert in experts:
                    if other_expert['name'] != expert['name']:
                        answer = self.generate_expert_message(
                            expert_name=other_expert['name'],
                            expert_domain=other_expert['domain'],
                            task_analysis=task_analysis,
                            message_type='answer',
                            context=round_messages[-5:]
                        )
                        round_messages.append(answer)

        # 4. Final critique
        critique = self.generate_expert_message(
            expert_name="Critic",
            expert_domain="Research Evaluation",
            task_analysis=task_analysis,
            message_type='critique',
            context=round_messages
        )
        round_messages.append(critique)

        self.current_round += 1
        return round_messages

    def extract_consensus_score(self, critique_message: DiscussionMessage) -> float:
        """Extract consensus score from critique message"""
        try:
            # Look for numerical scores in the critique
            import re
            numbers = re.findall(r'0\.\d+|1\.0', critique_message.content)
            if numbers:
                return float(numbers[0])
        except:
            pass
        return 0.7  # Default score if extraction fails

    def format_discussion_output(self, messages: List[DiscussionMessage]) -> Dict[str, Any]:
        """Format discussion messages into structured output"""
        output = {
            "round": self.current_round,
            "proposals": [],
            "feedback": [],
            "questions_and_answers": [],
            "critique": None,
            "consensus_score": 0.0
        }

        for msg in messages:
            if msg.message_type == 'proposal':
                output["proposals"].append({
                    "expert": msg.expert_name,
                    "content": msg.content
                })
            elif msg.message_type == 'feedback':
                output["feedback"].append({
                    "expert": msg.expert_name,
                    "content": msg.content,
                    "references": msg.references
                })
            elif msg.message_type in ['question', 'answer']:
                output["questions_and_answers"].append({
                    "expert": msg.expert_name,
                    "type": msg.message_type,
                    "content": msg.content,
                    "references": msg.references
                })
            elif msg.message_type == 'critique':
                output["critique"] = {
                    "content": msg.content,
                    "score": self.extract_consensus_score(msg)
                }
                output["consensus_score"] = output["critique"]["score"]

        return output

    def _format_perturbations(self, perturbations: List[Dict[str, Any]]) -> str:
        formatted = []
        for p in perturbations:
            desc = f"- Type: {p['type']}\n"
            if 'targets' in p:
                desc += f"  Targets: {', '.join(p['targets'])}\n"
            if 'description' in p and p['description']:
                desc += f"  Description: {p['description']}\n"
            formatted.append(desc)
        return '\n'.join(formatted)

    def _format_list(self, items: List[str]) -> str:

        return '\n'.join(f"- {item}" for item in items)

    def _get_relevant_knowledge(self, expert_domain: str, task_type: str, message_type: str) -> str:
        if not self.rag_retriever:
            return ""

        cache_key = f"{expert_domain}_{task_type}_{message_type}"

        if cache_key in self.knowledge_cache:
            return self.knowledge_cache[cache_key]

        query_terms = []

        domain_queries = {
            "deep_learning": [
                "neural network architecture",
                "deep learning model",
                "transformer model",
                "model optimization",
                task_type
            ],
            "single_cell_biology": [
                "single cell analysis",
                "cell type specific",
                "perturbation response",
                "pathway analysis",
                task_type
            ],
            "data_engineering": [
                "data preprocessing",
                "feature selection",
                "batch correction",
                "quality control",
                task_type
            ]
        }

        if expert_domain.lower() in domain_queries:
            query_terms.extend(domain_queries[expert_domain.lower()])

        message_type_queries = {
            "proposal": [
                "model architecture",
                "design pattern",
                "implementation strategy"
            ],
            "feedback": [
                "model evaluation",
                "performance analysis",
                "improvement strategy"
            ],
            "question": [
                "technical challenge",
                "implementation issue",
                "design consideration"
            ],
            "critique": [
                "performance metric",
                "evaluation criteria",
                "benchmark comparison"
            ]
        }

        if message_type in message_type_queries:
            query_terms.extend(message_type_queries[message_type])

        all_results = []
        for query in query_terms:
            try:
                results = self.rag_retriever.retrieve(query, top_k=3)
                all_results.extend(results)
            except Exception as e:
                print(f"Warning: Knowledge retrieval failed for query '{query}': {e}")

        if all_results:
            formatted_results = []
            seen_content = set()

            for result in all_results:
                content = (
                    result.get("abstract")
                    or result.get("content")
                    or result.get("snippet")
                    or ""
                )
                if content and content not in seen_content:
                    score = result.get("score", result.get("relevance_score", 0.0))
                    if score >= 0.5:
                        formatted_results.append(
                            f"Evidence ID: {result.get('evidence_id', 'untracked')}"
                        )
                        formatted_results.append(
                            f"Title: {result.get('title', 'Untitled')}"
                        )
                        formatted_results.append(
                            f"Source: {result.get('source', 'Unknown')}"
                        )
                        formatted_results.append(f"Relevance: {score:.2f}")
                        formatted_results.append(f"Content: {content}\n")
                        seen_content.add(content)

            knowledge_content = "\n".join(formatted_results)

            self.knowledge_cache[cache_key] = knowledge_content

            return knowledge_content

        return ""

    def _get_domain_context(self, expert_domain: str) -> str:
        domain_contexts = {
            "deep_learning": """
Key Considerations for Deep Learning in Single-Cell Analysis:
1. Architecture Design:
   - Transformer-based architectures for capturing cell-cell interactions
   - Graph neural networks for pathway integration
   - Multi-task learning for joint prediction of multiple genes
   - Attention mechanisms for perturbation-specific effects

2. Current SOTA Approaches:
   - scGPT: Transformer-based model for cell state prediction
   - GEARS: Graph-based model for gene expression prediction
   - CPA: Compositional perturbation autoencoder

3. Common Challenges:
   - High dimensionality of gene expression data
   - Sparsity and dropout in single-cell data
   - Limited samples for rare cell types
   - Complex cellular response mechanisms

4. Performance Metrics:
   - Mean squared error for expression levels
   - Pearson correlation for gene-level accuracy
   - Biological pathway enrichment scores
   - Cell type preservation metrics""",

            "single_cell_biology": """
Key Biological Considerations:
1. Cell Type Specificity:
   - Cell type-specific response patterns
   - Lineage relationships and developmental trajectories
   - Cell state transitions and plasticity
   - Tissue-specific regulatory networks

2. Perturbation Response:
   - Direct vs indirect effects
   - Temporal dynamics of responses
   - Compensatory mechanisms
   - Pathway cross-talk

3. Regulatory Networks:
   - Transcription factor networks
   - Signaling cascades
   - Metabolic pathways
   - Gene regulatory circuits

4. Quality Control:
   - Technical vs biological variation
   - Batch effects and normalization
   - Cell cycle effects
   - Dropout patterns""",

            "data_engineering": """
Key Data Engineering Considerations:
1. Data Processing:
   - Quality control metrics and thresholds
   - Normalization methods for single-cell data
   - Feature selection strategies
   - Batch effect correction

2. Data Integration:
   - Multi-omics data integration
   - Cross-dataset harmonization
   - Metadata incorporation
   - External knowledge base integration

3. Computational Efficiency:
   - Scalable data structures
   - Efficient matrix operations
   - Memory optimization
   - Parallel processing

4. Data Quality:
   - Missing value handling
   - Noise reduction
   - Outlier detection
   - Data validation procedures"""
        }

        return domain_contexts.get(expert_domain.lower(),
            "No specific domain context available for this expert type.")

    def _fill_missing_fields(self, task_analysis: Dict[str, Any], missing_fields: List[str]) -> Dict[str, Any]:
        task_analysis = task_analysis.copy()

        defaults = {
            'task_type': 'perturbation_prediction',
            'dataset': {
                'name': 'single_cell_dataset',
                'type': 'RNA-seq',
                'description': 'Single-cell RNA sequencing dataset'
            },
            'perturbations': [
                {
                    'type': 'gene_knockout',
                    'description': 'CRISPR-based gene knockout',
                    'targets': ['gene1', 'gene2']
                }
            ],
            'cell_types': ['T_cells', 'B_cells', 'monocytes'],
            'objectives': [
                'Predict cellular responses to perturbations',
                'Identify key regulatory genes',
                'Understand cellular heterogeneity'
            ],
            'constraints': [
                'Computational efficiency',
                'Biological interpretability',
                'Scalability to large datasets'
            ],
            'evaluation_metrics': [
                'Accuracy',
                'Precision',
                'Recall',
                'F1-score'
            ]
        }

        for field in missing_fields:
            if field in defaults:
                task_analysis[field] = defaults[field]
                print(f"Added default value for {field}")

        return task_analysis

    def _validate_llm_config(self):
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        config_status = self.llm_client.get_config_status()

        providers_configured = [
            k for k, v in config_status.items()
            if k.endswith('_configured') and v
        ]

        if not providers_configured:
            raise ValueError(
                "No LLM providers configured. Please set up at least one provider "
                "(OpenAI, Anthropic, DeepSeek, etc.) with valid API keys."
            )

        print(f"✅ LLM configured with providers: {', '.join(p.replace('_configured', '') for p in providers_configured)}")
        print(f"Using model: {config_status.get('model_name', 'default')}")

    def save_discussion_log(self, output_dir: str):
        """Save discussion history to file"""
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"discussion_round_{self.current_round}.json")

        discussion_log = {
            "total_rounds": self.current_round,
            "messages": [
                {
                    "expert": msg.expert_name,
                    "type": msg.message_type,
                    "content": msg.content,
                    "round": msg.round,
                    "references": msg.references,
                    "confidence_score": msg.confidence_score
                }
                for msg in self.discussion_history
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(discussion_log, f, indent=2, ensure_ascii=False)

        summary_file = os.path.join(output_dir, "discussion_summary.json")
        summary = {
            "total_rounds": self.current_round,
            "experts": list(set(msg.expert_name for msg in self.discussion_history)),
            "message_types": list(set(msg.message_type for msg in self.discussion_history)),
            "rounds_summary": [
                {
                    "round": i,
                    "messages": len([m for m in self.discussion_history if m.round == i]),
                    "experts_involved": list(set(m.expert_name for m in self.discussion_history if m.round == i))
                }
                for i in range(self.current_round + 1)
            ]
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
