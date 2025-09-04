import { LanguageModel, ReflectiveDataset } from '../types/index.js';

/**
 * DSPy-style structured instruction proposal
 * Implements sophisticated prompt engineering based on execution traces
 */

interface InstructionProposalSignature {
  instructions: string;
  currentInstruction: string;
  datasetWithFeedback: Array<Record<string, any>>;
  analysisPrompt: string;
}

export class DSPyInstructionProposer {
  private lm: LanguageModel;
  
  constructor(lm: LanguageModel) {
    this.lm = lm;
  }
  
  /**
   * Analyze execution traces to identify improvement opportunities
   */
  private analyzeTraces(examples: Array<Record<string, any>>): {
    failurePatterns: string[];
    successPatterns: string[];
    commonErrors: string[];
    recommendations: string[];
  } {
    const failurePatterns: string[] = [];
    const successPatterns: string[] = [];
    const commonErrors: Map<string, number> = new Map();
    const recommendations: Set<string> = new Set();
    
    for (const example of examples) {
      const score = example['Score'] || 0;
      const feedback = example['Feedback'] || '';
      
      if (score < 0.5) {
        // Failure case
        failurePatterns.push(feedback);
        
        // Extract specific error types
        if (feedback.includes('format')) {
          commonErrors.set('format_error', (commonErrors.get('format_error') || 0) + 1);
          recommendations.add('Specify output format explicitly');
        }
        if (feedback.includes('missing')) {
          commonErrors.set('missing_output', (commonErrors.get('missing_output') || 0) + 1);
          recommendations.add('Ensure all required fields are generated');
        }
        if (feedback.includes('incorrect')) {
          commonErrors.set('incorrect_logic', (commonErrors.get('incorrect_logic') || 0) + 1);
          recommendations.add('Clarify the reasoning process');
        }
      } else if (score > 0.8) {
        // Success case
        const outputs = example['Generated Outputs'];
        if (outputs) {
          successPatterns.push(JSON.stringify(outputs));
        }
      }
    }
    
    // Sort errors by frequency
    const sortedErrors = Array.from(commonErrors.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([error]) => error);
    
    return {
      failurePatterns: failurePatterns.slice(0, 5),
      successPatterns: successPatterns.slice(0, 3),
      commonErrors: sortedErrors,
      recommendations: Array.from(recommendations)
    };
  }
  
  /**
   * Generate structured prompt for instruction improvement
   */
  private buildProposalPrompt(
    currentInstruction: string,
    analysis: ReturnType<typeof this.analyzeTraces>,
    examples: Array<Record<string, any>>
  ): string {
    const prompt = `I am trying to solve a task using the DSPy framework. Here's a comprehensive overview of DSPy concepts to guide your improvements:

Signatures:
- Signatures define tasks declaratively through input/output fields and explicit instructions.
- They serve as blueprints for what the LM needs to accomplish.

Current Instruction:
"""
${currentInstruction}
"""

Execution Analysis:
- Average Score: ${this.calculateAverageScore(examples).toFixed(2)}
- Success Rate: ${this.calculateSuccessRate(examples).toFixed(2)}%
- Common Errors: ${analysis.commonErrors.join(', ') || 'None identified'}

Failure Patterns:
${analysis.failurePatterns.map((p, i) => `${i + 1}. ${p}`).join('\n') || 'No failures observed'}

Successful Output Examples:
${analysis.successPatterns.slice(0, 2).join('\n') || 'No successful patterns captured'}

Recommendations from Analysis:
${analysis.recommendations.map(r => `- ${r}`).join('\n') || 'No specific recommendations'}

Dataset Examples with Feedback:
${JSON.stringify(examples.slice(0, 3), null, 2)}

Assignment:
- Think step-by-step: First, deeply analyze the current instruction, traces, and feedback to identify failure modes, strengths, and opportunities.
- Create a concise checklist (3-7 bullets) outlining your high-level improvement plan.
- Then, propose an improved instruction that directly addresses the identified issues.
- The new instruction should be:
  * Self-contained and clear
  * Include specific guidance for edge cases
  * Address the common errors identified
  * Incorporate successful patterns
  * Be actionable without external context

Output the improved instruction only, without additional explanations:`;
    
    return prompt;
  }
  
  private calculateAverageScore(examples: Array<Record<string, any>>): number {
    if (examples.length === 0) return 0;
    const sum = examples.reduce((acc, ex) => acc + (ex['Score'] || 0), 0);
    return sum / examples.length;
  }
  
  private calculateSuccessRate(examples: Array<Record<string, any>>): number {
    if (examples.length === 0) return 0;
    const successes = examples.filter(ex => (ex['Score'] || 0) >= 0.5).length;
    return (successes / examples.length) * 100;
  }
  
  /**
   * Propose improved instruction based on reflective dataset
   */
  async proposeInstruction(
    currentInstruction: string,
    datasetWithFeedback: Array<Record<string, any>>
  ): Promise<string> {
    // Analyze the dataset
    const analysis = this.analyzeTraces(datasetWithFeedback);
    
    // Build proposal prompt
    const prompt = this.buildProposalPrompt(
      currentInstruction,
      analysis,
      datasetWithFeedback
    );
    
    // Get improved instruction from LM
    const response = await this.lm(prompt);
    
    // Clean and validate response
    let improvedInstruction = response.trim();
    
    // Remove any markdown or quotes if present
    improvedInstruction = improvedInstruction
      .replace(/^```[\w]*\n?/, '')
      .replace(/\n?```$/, '')
      .replace(/^["']|["']$/g, '');
    
    // Ensure minimum quality
    if (improvedInstruction.length < 20) {
      // Fallback to enhanced version of current instruction
      improvedInstruction = this.enhanceInstructionLocally(
        currentInstruction,
        analysis
      );
    }
    
    return improvedInstruction;
  }
  
  /**
   * Local enhancement when LM fails or returns poor quality
   */
  private enhanceInstructionLocally(
    currentInstruction: string,
    analysis: ReturnType<typeof this.analyzeTraces>
  ): string {
    let enhanced = currentInstruction;
    
    // Add error handling guidance if missing
    if (!enhanced.includes('error') && analysis.commonErrors.length > 0) {
      enhanced += '\n\nCommon pitfalls to avoid:\n';
      enhanced += analysis.commonErrors.map(err => {
        switch(err) {
          case 'format_error':
            return '- Ensure output follows the exact format specified';
          case 'missing_output':
            return '- Generate all required output fields';
          case 'incorrect_logic':
            return '- Apply logical reasoning step-by-step';
          default:
            return `- Address ${err.replace(/_/g, ' ')}`;
        }
      }).join('\n');
    }
    
    // Add recommendations
    if (analysis.recommendations.length > 0) {
      enhanced += '\n\nKey guidelines:\n';
      enhanced += analysis.recommendations.map(r => `- ${r}`).join('\n');
    }
    
    // Add success criteria if not present
    if (!enhanced.includes('success') && !enhanced.includes('quality')) {
      enhanced += '\n\nFor best results:\n';
      enhanced += '- Be precise and detailed in your response\n';
      enhanced += '- Show your reasoning when applicable\n';
      enhanced += '- Validate your output before finalizing';
    }
    
    return enhanced;
  }
  
  /**
   * Propose multiple instruction variants for A/B testing
   */
  async proposeVariants(
    currentInstruction: string,
    datasetWithFeedback: Array<Record<string, any>>,
    numVariants: number = 3
  ): Promise<string[]> {
    const variants: string[] = [];
    
    // Generate base improved instruction
    const baseImproved = await this.proposeInstruction(
      currentInstruction,
      datasetWithFeedback
    );
    variants.push(baseImproved);
    
    // Generate variations
    const analysis = this.analyzeTraces(datasetWithFeedback);
    
    // Variant 1: Focus on structure
    variants.push(this.createStructuredVariant(baseImproved, analysis));
    
    // Variant 2: Focus on examples
    if (analysis.successPatterns.length > 0) {
      variants.push(this.createExampleBasedVariant(baseImproved, analysis));
    }
    
    // Variant 3: Focus on constraints
    if (numVariants > 2) {
      variants.push(this.createConstraintFocusedVariant(baseImproved, analysis));
    }
    
    return variants.slice(0, numVariants);
  }
  
  private createStructuredVariant(
    base: string,
    analysis: ReturnType<typeof this.analyzeTraces>
  ): string {
    return `${base}

Structure your response as follows:
1. Initial analysis of the input
2. Step-by-step processing
3. Validation of output
4. Final formatted result`;
  }
  
  private createExampleBasedVariant(
    base: string,
    analysis: ReturnType<typeof this.analyzeTraces>
  ): string {
    return `${base}

Example of successful output pattern:
${analysis.successPatterns[0] || '{}'}

Ensure your output matches this structure.`;
  }
  
  private createConstraintFocusedVariant(
    base: string,
    analysis: ReturnType<typeof this.analyzeTraces>
  ): string {
    return `${base}

Critical constraints:
${analysis.recommendations.map(r => `✓ ${r}`).join('\n')}

Failure criteria (avoid these):
${analysis.commonErrors.map(e => `✗ ${e.replace(/_/g, ' ')}`).join('\n')}`;
  }
}

/**
 * Factory function for creating instruction proposer
 */
export function createInstructionProposer(lm: LanguageModel): DSPyInstructionProposer {
  return new DSPyInstructionProposer(lm);
}

/**
 * Standalone function for proposing instructions (compatible with existing API)
 */
export async function proposeInstructionWithDSPy(
  lm: LanguageModel,
  currentInstruction: string,
  datasetWithFeedback: Array<Record<string, any>>
): Promise<string> {
  const proposer = new DSPyInstructionProposer(lm);
  return proposer.proposeInstruction(currentInstruction, datasetWithFeedback);
}