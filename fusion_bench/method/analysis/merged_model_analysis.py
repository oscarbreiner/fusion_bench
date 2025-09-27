"""
Merged Model Analysis: Sign Conflicts, L2 Distance, and Cosine Similarity

This module provides comprehensive analysis of merged models by comparing their weights
with individual fine-tuned models across three key metrics:

1. Sign Conflicts: Element-wise sign comparison between merged and individual models
2. L2 Distance: Euclidean distance between flattened weight vectors  
3. Cosine Similarity: Cosine similarity between flattened weight vectors

The analysis supports all major model merging methods:
- FastFood merging
- EMR merging  
- Task Arithmetic
- TIES merging
- AdaMerging
- Simple Average

Based on the framework of fusion_bench analysis tools.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import LightningFabricMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.parameters import (
    StateDictType,
    state_dict_to_vector,
    trainable_state_dict,
)
from fusion_bench.utils.state_dict_arithmetic import state_dict_sub

# Import TaskSpecificEvaluationWrapper for handling enhanced EMR task-specific analysis
try:
    from fusion_bench.method.enhanced_emr_merging import TaskSpecificEvaluationWrapper
except ImportError:
    TaskSpecificEvaluationWrapper = None

log = logging.getLogger(__name__)


@auto_register_config
class MergedModelAnalysis(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Comprehensive analysis of merged model weights compared to individual fine-tuned models.
    
    This algorithm computes three key metrics between merged models and individual models:
    
    1. **Sign Conflicts**: Element-wise comparison of weight signs
       - Calculates the ratio of elements with conflicting signs
       - Reports average across all individual models
       
    2. **L2 Distance**: Euclidean distance between flattened weight vectors
       - Flattens all model weights into 1D vectors
       - Computes L2 norm of the difference
       - Reports average across all individual models
       
    3. **Cosine Similarity**: Cosine similarity between flattened weight vectors
       - Measures angular similarity between weight vectors
       - Reports average across all individual models
    
    Args:
        merging_methods (List[str]): List of merging methods to analyze
        trainable_only (bool): Whether to only analyze trainable parameters
        output_path (str, optional): Directory to save analysis results
        save_individual_results (bool): Whether to save per-model detailed results
        
    Outputs:
        - merged_model_analysis_summary.csv: Summary statistics for all methods
        - merged_model_analysis_detailed.json: Detailed per-model results
        - merged_model_analysis_report.txt: Human-readable analysis report
        - merged_model_analysis_plots.pdf: Visualization with three stacked bar plots
    """

    _config_mapping = BaseAlgorithm._config_mapping | {
        "_output_path": "output_path",
    }

    def __init__(
        self,
        merging_methods: List[str],
        trainable_only: bool = True,
        output_path: Optional[str] = None,
        save_individual_results: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.merging_methods = merging_methods
        self.trainable_only = trainable_only
        self._output_path = output_path
        self.save_individual_results = save_individual_results
        
        # Validate merging methods
        supported_methods = {
            'fastfood_merging', 'emr_merging', 'emr_merging_enhanced', 'task_arithmetic', 
            'ties_merging', 'adamerging', 'simple_average', 'regmean', 'regmean_plusplus'
        }
        for method in self.merging_methods:
            if method not in supported_methods:
                log.warning(f"Method '{method}' may not be supported. Supported: {supported_methods}")

    @property
    def output_path(self):
        if self._output_path is None:
            return self.fabric.logger.log_dir
        else:
            return self._output_path

    def _get_model_state_dict(self, model: nn.Module) -> StateDictType:
        """Get state dict from model, optionally filtering to trainable parameters only."""
        if self.trainable_only:
            return trainable_state_dict(model)
        else:
            return model.state_dict()

    def _flatten_state_dict(self, state_dict: StateDictType) -> torch.Tensor:
        """Flatten state dict into a 1D tensor."""
        return state_dict_to_vector(state_dict)

    def _compute_sign_conflicts(self, merged_weights: torch.Tensor, individual_weights: torch.Tensor) -> float:
        """
        Compute the ratio of elements with conflicting signs.
        
        Args:
            merged_weights: Flattened merged model weights
            individual_weights: Flattened individual model weights
            
        Returns:
            Ratio of elements with conflicting signs (0.0 to 1.0)
        """
        # Ensure tensors are on the same device
        if merged_weights.device != individual_weights.device:
            individual_weights = individual_weights.to(merged_weights.device)
        
        # Get signs of weights
        merged_signs = torch.sign(merged_weights)
        individual_signs = torch.sign(individual_weights)
        
        # Count conflicts (different signs, excluding zeros)
        non_zero_mask = (merged_signs != 0) & (individual_signs != 0)
        if non_zero_mask.sum() == 0:
            return 0.0
            
        conflicts = (merged_signs[non_zero_mask] != individual_signs[non_zero_mask])
        conflict_ratio = conflicts.float().mean().item()
        
        return conflict_ratio

    def _compute_l2_distance(self, merged_weights: torch.Tensor, individual_weights: torch.Tensor) -> float:
        """
        Compute L2 (Euclidean) distance between weight vectors.
        
        Args:
            merged_weights: Flattened merged model weights
            individual_weights: Flattened individual model weights
            
        Returns:
            L2 distance between the weight vectors
        """
        # Ensure tensors are on the same device
        if merged_weights.device != individual_weights.device:
            individual_weights = individual_weights.to(merged_weights.device)
        
        return torch.norm(merged_weights - individual_weights, p=2).item()

    def _compute_cosine_similarity(self, merged_weights: torch.Tensor, individual_weights: torch.Tensor) -> float:
        """
        Compute cosine similarity between weight vectors.
        
        Args:
            merged_weights: Flattened merged model weights
            individual_weights: Flattened individual model weights
            
        Returns:
            Cosine similarity between the weight vectors (-1.0 to 1.0)
        """
        # Ensure tensors are on the same device
        if merged_weights.device != individual_weights.device:
            individual_weights = individual_weights.to(merged_weights.device)
        
        # Compute cosine similarity directly (F.cosine_similarity handles normalization internally)
        similarity = F.cosine_similarity(merged_weights.unsqueeze(0), individual_weights.unsqueeze(0), dim=1)
        return similarity.item()

    def _analyze_task_specific_method(
        self, 
        method_name: str,
        task_wrapper: 'TaskSpecificEvaluationWrapper', 
        individual_models: Dict[str, nn.Module]
    ) -> Dict[str, Any]:
        """
        Analyze a task-specific method (Enhanced EMR with TaskSpecificEvaluationWrapper).
        
        Args:
            method_name: Name of the merging method
            task_wrapper: TaskSpecificEvaluationWrapper containing task-specific models
            individual_models: Dictionary of individual fine-tuned models
            
        Returns:
            Dictionary containing analysis results with task-specific breakdowns
        """
        log.info(f"Analyzing task-specific method: {method_name}")
        
        results = {
            'method': method_name,
            'task_specific_results': {},  # Per-task analysis
            'individual_results': {},     # Overall individual results
            'summary': {},
            'task_summary': {}           # Per-task summaries
        }
        
        all_sign_conflicts = []
        all_l2_distances = []
        all_cosine_similarities = []
        
        # Analyze each task-specific model against its corresponding individual model
        for task_name, task_model in task_wrapper.task_models.items():
            if task_name not in individual_models:
                log.warning(f"Task '{task_name}' not found in individual models, skipping")
                continue
                
            log.info(f"Analyzing task-specific model for task: {task_name}")
            
            # Get task-specific model weights  
            task_state_dict = self._get_model_state_dict(task_model)
            task_weights = self._flatten_state_dict(task_state_dict)
            
            # Get corresponding individual model weights
            individual_model = individual_models[task_name]
            individual_state_dict = self._get_model_state_dict(individual_model)
            individual_weights = self._flatten_state_dict(individual_state_dict)
            
            # Ensure same device
            target_device = task_weights.device
            if individual_weights.device != target_device:
                individual_weights = individual_weights.to(target_device)
            
            # Compute metrics for this specific task
            sign_conflict = self._compute_sign_conflicts(task_weights, individual_weights)
            l2_distance = self._compute_l2_distance(task_weights, individual_weights)
            cosine_similarity = self._compute_cosine_similarity(task_weights, individual_weights)
            
            # Store task-specific results
            results['task_specific_results'][task_name] = {
                'sign_conflict_ratio': sign_conflict,
                'l2_distance': l2_distance, 
                'cosine_similarity': cosine_similarity
            }
            
            # Also store in individual results for compatibility
            results['individual_results'][task_name] = {
                'sign_conflict_ratio': sign_conflict,
                'l2_distance': l2_distance,
                'cosine_similarity': cosine_similarity
            }
            
            # Collect for overall averaging
            all_sign_conflicts.append(sign_conflict)
            all_l2_distances.append(l2_distance)
            all_cosine_similarities.append(cosine_similarity)
            
            log.info(f"Task {task_name} - Sign conflicts: {sign_conflict:.4f}, "
                    f"L2 distance: {l2_distance:.4f}, Cosine sim: {cosine_similarity:.4f}")
        
        # Compute overall summary statistics
        if all_sign_conflicts:
            results['summary'] = {
                'avg_sign_conflict_ratio': np.mean(all_sign_conflicts),
                'std_sign_conflict_ratio': np.std(all_sign_conflicts),
                'avg_l2_distance': np.mean(all_l2_distances),
                'std_l2_distance': np.std(all_l2_distances),
                'avg_cosine_similarity': np.mean(all_cosine_similarities),
                'std_cosine_similarity': np.std(all_cosine_similarities),
                'num_models_analyzed': len(all_sign_conflicts)  # Use consistent key name
            }
            
            # Per-task summary for detailed analysis
            results['task_summary'] = {
                'best_task_cosine_sim': max(all_cosine_similarities),
                'worst_task_cosine_sim': min(all_cosine_similarities),
                'best_task_l2_dist': min(all_l2_distances), 
                'worst_task_l2_dist': max(all_l2_distances),
                'task_performance_variance': np.var(all_cosine_similarities)
            }
            
            log.info(f"Task-specific analysis complete - {len(all_sign_conflicts)} tasks analyzed")
        
        return results

    def _analyze_method(
        self, 
        method_name: str,
        merged_model: nn.Module, 
        individual_models: Dict[str, nn.Module]
    ) -> Dict[str, Any]:
        """
        Analyze a single merging method.
        
        Args:
            method_name: Name of the merging method
            merged_model: The merged model
            individual_models: Dictionary of individual fine-tuned models
            
        Returns:
            Dictionary containing analysis results
        """
        log.info(f"Analyzing method: {method_name}")
        
        # Get merged model weights
        merged_state_dict = self._get_model_state_dict(merged_model)
        merged_weights = self._flatten_state_dict(merged_state_dict)
        
        # Determine target device from merged model weights
        target_device = merged_weights.device
        log.debug(f"Target device for analysis: {target_device}")
        
        results = {
            'method': method_name,
            'individual_results': {},
            'summary': {}
        }
        
        sign_conflicts = []
        l2_distances = []
        cosine_similarities = []
        
        # Analyze against each individual model
        for model_name, individual_model in tqdm(individual_models.items(), 
                                               desc=f"Analyzing {method_name}"):
            # Get individual model weights
            individual_state_dict = self._get_model_state_dict(individual_model)
            individual_weights = self._flatten_state_dict(individual_state_dict)
            
            # Ensure weights are on the same device as merged weights
            if individual_weights.device != target_device:
                log.debug(f"Moving {model_name} weights from {individual_weights.device} to {target_device}")
                individual_weights = individual_weights.to(target_device)
            
            # Ensure same size (should be the case for same architecture)
            if merged_weights.shape != individual_weights.shape:
                log.warning(f"Shape mismatch for {model_name}: merged {merged_weights.shape} vs individual {individual_weights.shape}")
                continue
            
            # Compute metrics
            sign_conflict = self._compute_sign_conflicts(merged_weights, individual_weights)
            l2_distance = self._compute_l2_distance(merged_weights, individual_weights)
            cosine_similarity = self._compute_cosine_similarity(merged_weights, individual_weights)
            
            # Store individual results
            results['individual_results'][model_name] = {
                'sign_conflict_ratio': sign_conflict,
                'l2_distance': l2_distance,
                'cosine_similarity': cosine_similarity
            }
            
            # Collect for averaging
            sign_conflicts.append(sign_conflict)
            l2_distances.append(l2_distance)
            cosine_similarities.append(cosine_similarity)
            
            log.debug(f"{model_name} - Sign conflicts: {sign_conflict:.4f}, "
                     f"L2 distance: {l2_distance:.4f}, Cosine sim: {cosine_similarity:.4f}")
        
        # Compute summary statistics
        if sign_conflicts:  # Only if we have results
            results['summary'] = {
                'avg_sign_conflict_ratio': np.mean(sign_conflicts),
                'std_sign_conflict_ratio': np.std(sign_conflicts),
                'avg_l2_distance': np.mean(l2_distances),
                'std_l2_distance': np.std(l2_distances),
                'avg_cosine_similarity': np.mean(cosine_similarities),
                'std_cosine_similarity': np.std(cosine_similarities),
                'num_models_analyzed': len(sign_conflicts)
            }
            
            log.info(f"{method_name} Summary:")
            log.info(f"  Avg Sign Conflict Ratio: {results['summary']['avg_sign_conflict_ratio']:.4f} ± {results['summary']['std_sign_conflict_ratio']:.4f}")
            log.info(f"  Avg L2 Distance: {results['summary']['avg_l2_distance']:.4f} ± {results['summary']['std_l2_distance']:.4f}")
            log.info(f"  Avg Cosine Similarity: {results['summary']['avg_cosine_similarity']:.4f} ± {results['summary']['std_cosine_similarity']:.4f}")
        
        return results

    def _analyze_emr_enhanced_method(
        self,
        method_name: str,
        emr_wrapper,  # TaskSpecificEvaluationWrapper
        individual_models: Dict[str, nn.Module]
    ) -> Dict[str, Any]:
        """
        Analyze EMR Enhanced method with task-specific routing.
        
        For EMR Enhanced, we need to analyze each task-specific model against
        its corresponding individual model, since EMR uses parameter routing.
        
        Args:
            method_name: Name of the merging method (should be 'emr_merging_enhanced')
            emr_wrapper: TaskSpecificEvaluationWrapper from Enhanced EMR
            individual_models: Dictionary of individual fine-tuned models
            
        Returns:
            Dictionary containing analysis results
        """
        log.info(f"Analyzing EMR Enhanced method: {method_name}")
        
        # Check if this is actually a TaskSpecificEvaluationWrapper
        if TaskSpecificEvaluationWrapper and isinstance(emr_wrapper, TaskSpecificEvaluationWrapper):
            log.info("Using task-specific analysis for Enhanced EMR")
            return self._analyze_task_specific_method(method_name, emr_wrapper, individual_models)
        else:
            log.warning(f"EMR Enhanced expected TaskSpecificEvaluationWrapper, got {type(emr_wrapper)}")
            # Fall back to regular analysis
            return self._analyze_method(method_name, emr_wrapper, individual_models)

    def _load_method_with_config(self, method_name: str):
        """
        Load method algorithm using actual Hydra configuration files.
        
        Args:
            method_name: Name of the merging method
            
        Returns:
            Instantiated algorithm or None if config not found
        """
        from omegaconf import OmegaConf
        from fusion_bench.utils import instantiate
        import os
        
        # Get the config path from the current environment
        config_dir = None
        for potential_config_dir in [
            "config/method",  # Relative to current directory
            "fusion_bench/config/method",  # Relative to repo root
            os.path.join(os.path.dirname(__file__), "../../config/method"),  # Relative to this file
            os.path.join(os.getcwd(), "fusion_bench/config/method"),  # From current working directory
            "/dss/dsshome1/08/di97hih/fusion_merging/fusion_bench/config/method",  # Absolute path fallback
        ]:
            if os.path.exists(potential_config_dir):
                config_dir = potential_config_dir
                log.debug(f"Found config directory: {config_dir}")
                break
        
        if config_dir is None:
            log.warning(f"Could not find config directory for method {method_name}")
            return None
        
        # Try to load the configuration file
        config_file = os.path.join(config_dir, f"{method_name}.yaml")
        if not os.path.exists(config_file):
            log.warning(f"Config file not found: {config_file}")
            return None
            
        try:
            # Load the YAML configuration
            config = OmegaConf.load(config_file)
            log.info(f"Loading {method_name} with config: {OmegaConf.to_yaml(config)}")
            
            # Instantiate the algorithm using the configuration
            algorithm = instantiate(config)
            return algorithm
            
        except Exception as e:
            log.warning(f"Failed to load config for {method_name}: {e}")
            return None

    def _load_merged_model(self, method_name: str, modelpool: BaseModelPool) -> Optional[nn.Module]:
        """
        Load a merged model using the specified method with actual Hydra configuration.
        
        Args:
            method_name: Name of the merging method
            modelpool: Model pool containing the models to merge
            
        Returns:
            Merged model or None if method not supported
        """
        try:
            # Load the actual Hydra configuration for this method
            algorithm = self._load_method_with_config(method_name)
            if algorithm is None:
                log.error(f"Failed to load algorithm for {method_name} using Hydra config")
                return None
            
            log.info(f"Running {method_name} algorithm...")
            
            # Run the merging algorithm
            merged_model = algorithm.run(modelpool)
            
            # Handle case where the returned model has compatibility issues
            if hasattr(merged_model, 'config') and hasattr(merged_model.config, 'return_dict'):
                # Remove problematic config attribute if it exists
                if 'return_dict' in merged_model.config.__dict__:
                    delattr(merged_model.config, 'return_dict')
            
            # Handle case where algorithm returns list of models
            if isinstance(merged_model, list):
                if len(merged_model) > 0:
                    merged_model = merged_model[0]  # Use first model
                    log.info(f"Algorithm returned list of {len(merged_model)} models, using first")
                else:
                    log.error(f"Algorithm returned empty list for {method_name}")
                    return None
                    
            # Handle TaskSpecificEvaluationWrapper for enhanced EMR analysis
            if TaskSpecificEvaluationWrapper and isinstance(merged_model, TaskSpecificEvaluationWrapper):
                log.info(f"Detected task-specific model for {method_name}, enabling per-task analysis")
                # Return the wrapper - we'll handle task-specific analysis in the main analysis loop
                return merged_model
                    
            log.info(f"Successfully loaded merged model for {method_name}")
            return merged_model
            
        except ImportError as e:
            log.error(f"Import error for {method_name}: {e}")
            return None
        except Exception as e:
            log.error(f"Failed to load merged model for {method_name}: {e}")
            import traceback
            log.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        """
        Execute the comprehensive merged model analysis.
        
        Args:
            modelpool: Model pool containing pretrained and fine-tuned models
            
        Returns:
            Dictionary containing analysis results for all methods
        """
        log.info("Starting comprehensive merged model analysis")
        log.info(f"Methods to analyze: {self.merging_methods}")
        log.info(f"Trainable parameters only: {self.trainable_only}")
        
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load individual fine-tuned models
        individual_models = {}
        for model_name in modelpool.model_names:
            if model_name != '_pretrained_':
                model = modelpool.load_model(model_name)
                individual_models[model_name] = model
                
        log.info(f"Loaded {len(individual_models)} individual models: {list(individual_models.keys())}")
        
        # Analyze each merging method
        all_results = {}
        summary_data = []
        
        for method_name in self.merging_methods:
            log.info(f"Processing method: {method_name}")
            
            # Load merged model
            merged_model = self._load_merged_model(method_name, modelpool)
            if merged_model is None:
                log.warning(f"Skipping {method_name} due to loading failure")
                continue
            
            # Analyze the method - use specialized EMR analysis if needed
            if method_name == 'emr_merging_enhanced':
                # EMR Enhanced returns TaskSpecificEvaluationWrapper - needs special handling
                method_results = self._analyze_emr_enhanced_method(method_name, merged_model, individual_models)
            else:
                # Standard analysis for other methods
                method_results = self._analyze_method(method_name, merged_model, individual_models)
            all_results[method_name] = method_results
            
            # Add to summary data
            if 'summary' in method_results and method_results['summary']:
                summary_data.append({
                    'method': method_name,
                    **method_results['summary']
                })
        
        # Save results
        self._save_results(all_results, summary_data)
        
        log.info("Merged model analysis completed successfully")
        return all_results

    def _save_results(self, all_results: Dict[str, Any], summary_data: List[Dict[str, Any]]):
        """Save analysis results to files."""
        
        # Save summary CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_path, 'merged_model_analysis_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            log.info(f"Saved summary results to {summary_path}")
            
            # Print summary table
            log.info("Analysis Summary:")
            log.info("\n" + summary_df.to_string(index=False))
        
        # Save detailed JSON
        if self.save_individual_results:
            detailed_path = os.path.join(self.output_path, 'merged_model_analysis_detailed.json')
            with open(detailed_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            log.info(f"Saved detailed results to {detailed_path}")
        
        # Save human-readable report
        report_path = os.path.join(self.output_path, 'merged_model_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("Merged Model Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Configuration:\n")
            f.write(f"- Methods analyzed: {', '.join(self.merging_methods)}\n")
            f.write(f"- Trainable parameters only: {self.trainable_only}\n")
            f.write(f"- Output directory: {self.output_path}\n\n")
            
            for method_name, results in all_results.items():
                f.write(f"Method: {method_name}\n")
                f.write("-" * 30 + "\n")
                
                if 'summary' in results and results['summary']:
                    summary = results['summary']
                    # Handle both num_models_analyzed and num_tasks_analyzed
                    num_analyzed = summary.get('num_models_analyzed', summary.get('num_tasks_analyzed', 'Unknown'))
                    f.write(f"Models analyzed: {num_analyzed}\n")
                    f.write(f"Sign conflict ratio: {summary['avg_sign_conflict_ratio']:.4f} ± {summary['std_sign_conflict_ratio']:.4f}\n")
                    f.write(f"L2 distance: {summary['avg_l2_distance']:.4f} ± {summary['std_l2_distance']:.4f}\n")
                    f.write(f"Cosine similarity: {summary['avg_cosine_similarity']:.4f} ± {summary['std_cosine_similarity']:.4f}\n")
                else:
                    f.write("No results available\n")
                f.write("\n")
                
        log.info(f"Saved analysis report to {report_path}")
        
        # Create and save analysis plots
        self._create_analysis_plot(all_results)

    def _create_analysis_plot(self, all_results: Dict[str, Any]):
        """Create and save a comprehensive analysis plot with three stacked bar plots."""
        if not all_results:
            log.warning("No results to plot")
            return
            
        # Set seaborn style and color palette
        sns.set_style("whitegrid")
        colors = sns.color_palette("tab10", n_colors=len(self.merging_methods))
        
        # Prepare data for plotting
        methods = []
        tasks = []
        sign_conflicts = []
        l2_distances = []
        cosine_similarities = []
        
        # Collect individual task results
        for method_name, results in all_results.items():
            if 'individual_results' not in results:
                continue
                
            for task_name, task_results in results['individual_results'].items():
                methods.append(method_name)
                tasks.append(task_name)
                sign_conflicts.append(task_results['sign_conflict_ratio'])
                l2_distances.append(task_results['l2_distance'])
                cosine_similarities.append(task_results['cosine_similarity'])
        
        # Add average results
        for method_name, results in all_results.items():
            if 'summary' not in results or not results['summary']:
                continue
                
            summary = results['summary']
            methods.append(method_name)
            tasks.append('Average')
            sign_conflicts.append(summary['avg_sign_conflict_ratio'])
            l2_distances.append(summary['avg_l2_distance'])
            cosine_similarities.append(summary['avg_cosine_similarity'])
        
        # Check if we have data to plot
        if not methods:
            log.warning("No data collected for plotting")
            return
            
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Method': methods,
            'Task': tasks,
            'Sign_Conflict': sign_conflicts,
            'L2_Distance': l2_distances,
            'Cosine_Similarity': cosine_similarities
        })
        
        # Create figure with three subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle("Merged Model vs. Finetuned Task Models", fontsize=16, fontweight='bold')
        
        # Plot 1: Sign Conflict Ratio
        sns.barplot(data=plot_data, x='Task', y='Sign_Conflict', hue='Method', 
                   ax=axes[0], palette=colors)
        axes[0].set_title('Sign Conflict Ratio by Task and Method', fontweight='bold')
        axes[0].set_ylabel('Sign Conflict Ratio')
        axes[0].set_xlabel('')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: L2 Distance
        sns.barplot(data=plot_data, x='Task', y='L2_Distance', hue='Method', 
                   ax=axes[1], palette=colors)
        axes[1].set_title('L2 Distance by Task and Method', fontweight='bold')
        axes[1].set_ylabel('L2 Distance')
        axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Cosine Similarity
        sns.barplot(data=plot_data, x='Task', y='Cosine_Similarity', hue='Method', 
                   ax=axes[2], palette=colors)
        axes[2].set_title('Cosine Similarity by Task and Method', fontweight='bold')
        axes[2].set_ylabel('Cosine Similarity')
        axes[2].set_xlabel('Task')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legends
        
        # Save the plot
        plot_path = os.path.join(self.output_path, 'merged_model_analysis_plots.pdf')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        log.info(f"Saved analysis plots to {plot_path}")


# Configuration mapping for Hydra
MergedModelAnalysis._config_mapping = MergedModelAnalysis._config_mapping
