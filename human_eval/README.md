## Human Evaluation

### Annotation Tool
We used [**Label Studio**](https://labelstud.io/) for the annotation of our project. Label Studio is an open-source labeling platform that supports various machine learning annotation tasks with customizable and user-friendly patterns and interfaces.

During the annotation, we also provided information such as the number of words in each text, the primary patent IPC domain of the patent, and highlighted words that did not appear in the input texts. More details can be checked in the preprocessed JSON files that were uploaded to the platform for starting annotation:

- claims-to-abstract [`human_eval/preprocessed_for_annotation/human_eval_c2a.json`](https://github.com/ZoeYou/PatentEval/blob/main/human_eval/preprocessed_for_annotation/human_eval_c2a.json)
- next-claim generation [`human_eval/preprocessed_for_annotation/human_eval_c2c.json`](https://github.com/ZoeYou/PatentEval/blob/main/human_eval/preprocessed_for_annotation/human_eval_c2c.json)

### Annotated Dataset

We show our final annotations by exporting the JSON files directly from Label Studio:

[`human_eval/annotations/final_c2a.json`](https://github.com/ZoeYou/PatentEval/blob/main/human_eval/annotations/final_c2a.json) for claims-to-abstract generation, and [`human_eval/annotations/final_c2c.json`](https://github.com/ZoeYou/PatentEval/blob/main/human_eval/annotations/final_c2c.json) for next-claim generation.

### Annotation Guidelines
#### TASK 1: claims to abstract generation
**Task description**: Your task involves evaluating the quality of abstracts based on the information given in claims. The objective is to determine, each time you are presented with two generated abstracts (whether by machines or humans), which one exhibits higher quality. Additionally, please indicate the types of errors present in each output.

The errors can be categories into the following types:

- Grammatical Errors: Occurrences of incorrect grammar, punctuation, or sentence structure, including hallucinated repetitive sequences produced by language models.

- Irrelevant Content: Introducing content that deviates or digresses from the primary subject matter of the patent claims.

- Incomplete Coverage: Occurrences where the abstract omits essential components or concepts, failing to encapsulate all key points from the patent claims, especially the main (first independent) claim.

- Overly Wordy or Lengthy: Abstracts falling into this error type are not succinct, containing unnecessary details. Jurisdictions often impose word limits on abstracts — for example, in many English-speaking countries, abstracts are typically restricted to 150 words.

- Contradictory Information: Instances when the abstract introduces factual details that contradict the content found in the original claims.

- Unclarity: The abstract contains vague or ambiguous descriptions, making it difficult to grasp the intended message or details.

- Ineffective Summarization: Relates to abstracts that inadequately summarize the invention, often replicating one or more of the claims verbatim instead of providing a concise and comprehensive overview of the patent.



#### TASK 2: next claim generation
**Task description**: Your task involves evaluating the quality of the one newly generated next claim based on previous claims. The objective is to determine, each time you are presented with two generated claims (whether by machines or humans), which one exhibits higher quality. Additionally, please indicate the types of errors present in each output.

The errors can be categories into the following types:

- Grammatical Errors:
    - Grammatical Inaccuracy: Misuse of grammar and hallucinated repetitive sequences produced by language models.
    - Punctuation Discrepancy: Incorrect or inconsistent use of punctuation marks, deviating from standard patent drafting conventions.
    - Excessive Parentheticals: Over-reliance on parentheses for non-essential information, potentially detracting from claim clarity.

- Formatting Errors:
    - Claim Numbering Error: Incorrect or inconsistent numbering of claims.
    - Preamble Inconsistency Error: Inaccurate reflection of subject matter in the preamble, disrupting the conceptual flow between independent and dependent claims.
    - Transitional Phrase Error: Improper use of transitional phrases, impacting the scope of the claim.
    - Claim Body Disconnection: Presence of fewer than two elements or a lack of a coherent, logical connection between listed elements in the claim body.

- Dependency Errors:
    - Non-compliant Dependency with instruction: Dependency of the claims not matching the required dependency as instructed.
    - Dependency Clarity Error: Utilization of unclear multiple dependencies or an incorrect singular dependency.
    - Broad Scope Dependent Claims: Dependent claims that insufficiently narrow the scope of the independent claim they depend on.
    - Insufficient Differentiation of Independent Claims:  Independent claims that cover the same or similar scope as previous claims.

- Clarity Errors:
    - Vagueness: Usage of ambiguous, vague, or relative terms or expressions that render the claim's scope indefinite.
    - Antecedent Reference Errors: Failure to provide a clear antecedent basis for each term.
    - Terminological Inconsistency: Use of multiple terms or different reference numerals for the same element.
    - Wishful Claiming: Claims that express objectives without concrete methods, leading to speculative or abstract language.

- Brevity Errors:
    - Verbose Redundancy: Excessive wordiness without adding substantive content.
    - Sub-Optimal Claim Structure: Claims with complex language that could be more clearly expressed as multiple, simpler claims.

- Content Relevance Errors:
    - Irrelevant Matter Introduction: Introduction of matter unrelated to the disclosed embodiments, potentially broadening the claim beyond the invention's scope.

- Effectiveness Error:
    - Contradictory Claims: Claims that conflict with previous claims or do not follow a logical flow themselves.
    - Non-Distinctive Claim Repetition: Claims that lack effectiveness, primarily repeating content from earlier claims without adding new scope or detail.

By assessing these dimensions, you will help ensure a comprehensive evaluation of the claim quality and provide valuable insights into the errors present in each output.
