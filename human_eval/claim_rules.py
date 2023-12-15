# rules for checking the validity of a generated claim given previous claims
import re


class Rule_based_checker(object):
    def __init__(self, input_claims, generated_claim, required_dependent=True):
        self.input_claims = input_claims
        self.generated_claim = generated_claim
        self.required_dependent = required_dependent
        self.trans_phrases = ["further_comprising", "further_configured_to", "comprising", "consisting_of", "wherein", "in_which", "whereby", "such_that", "so_as_to", "characterized_in_that"]

        # extract the numberings of the input claims
        self.input_numberings = re.findall(r'[\s\n]?(\d+)[.)] ', self.input_claims)
        self.input_numberings = [int(numbering) for numbering in self.input_numberings if numbering != '']

    def numbering_coherence(self):
        # check if the numbering of the generated claim is coherent with the input claims
        # return True if the numbering is coherent, False otherwise
        # e.g. input claims: [1, 2, 3, 4, 5], generated claim: 6, return True
        # e.g. input claims: [1, 2, 3, 4, 5], generated claim: 7, return False

        # extract the numbering of the generated claim
        generated_numbering = re.findall(r'[\s\n]?(\d+)[.)] ', self.generated_claim)
        if len(generated_numbering) == 0:
            return False
        generated_numbering = int(generated_numbering[0]) if generated_numbering[0] != '' else int(generated_numbering[1])

        # check if the numbering of the generated claim is coherent with the input claims
        assert len(self.input_claims) != 0 and len(self.input_numberings) != 0

        if generated_numbering != self.input_numberings[-1] + 1:
            return False
        return True
    

    def depedency_correctness(self):
        # check if the dependency of the generated claim is correct as required
        # return True if the dependency is correct, False otherwise

        # extract the dependency of the generated claim
        is_dependent = re.search(r'claim \d+|claims', self.generated_claim)

        if self.required_dependent and is_dependent is None:
            return False
        if not self.required_dependent and is_dependent is not None:
            return False
        
        if is_dependent:
            # extract the numbering of the generated claim
            dependent_claim_numbering = re.findall(r'claims? (\d+)', self.generated_claim)
            try:
                dependent_claim_numbering = int(dependent_claim_numbering[0])
            except IndexError:
                if " any preceding claims" in self.generated_claim:
                    return True
                else:
                    return False

            # check if the dependency of the generated claim is in the input claims
            if dependent_claim_numbering not in self.input_numberings:
                return False

        return True
    
    
    def _befaft(self, s, words):  # splits at first occurrence of word from words
        reSplit = re.compile(
            r'(\b'+r'\b|\b'.join([w.replace('_', ' ') for w in words.split()])+r'\b)', re.I)
        return [t.strip() for t in (reSplit.split(s, maxsplit=1)+['', ''])[:3]]


    def punctuations_correctness(self):
        # check if the punctuations of the generated claim is correct as required
        # return True if the punctuations is correct, False otherwise

        # extract the punctuations of the generated claim
        punctuations = re.findall(r'[,.?!]', self.generated_claim)
        if len(punctuations) == 0:
            return False
        if punctuations[-1] != '.':
            return False

        # split by list in trans_phrases
        before, _, _ = self._befaft(self.generated_claim, " ".join(self.trans_phrases))
        

        if len(before) > 0 and before.strip()[-1] != ',':
            return False
        return True
    

    def parenthesis_correctness(self):
        # check if is the case that only numbers are in parenthesis
        # return True if the parenthesis is correct, False otherwise

        # extract the parenthesis of the generated claim
        parenthesis = re.findall(r'\((.*?)\)', self.generated_claim)
        if len(parenthesis) == 0:
            True
        for p in parenthesis:
            if not p.isdigit():
                return False
        return True


    def _remove_repetitive_spans(self):
        # Define a regular expression pattern to find repetitive spans of at least 2 repetitions.
        pattern = r'(?<![A-Za-z0-9])(.{2,}?)\1{1,}(?![A-Za-z0-9])'

        # Use re.sub to remove all matching repetitive spans in the text, except for the first instance.
        first_instance = re.sub(pattern, r'\1', self.generated_claim)

        return first_instance


    def no_hallucination(self):
        # check if the generated claim is hallucinated (repetition of phrases over three times)
        # return True if the generated claim is not hallucinated, False otherwise

        # remove repetitive spans
        first_instance = self._remove_repetitive_spans()
        # check if the generated claim is hallucinated
        if first_instance == self.generated_claim:
            return True
        return False
    

    def distinctive_claim(self):
        # check if the generated claim is distinctive
        # return True if the generated claim is distinctive, False otherwise

        # remove numbering of the beginning of the generated claim
        generated_claim = re.sub(r'^(\d+)[.)] ', '', self.generated_claim).rstrip(" \n.")

        # split input claims into sentences
        input_claims = re.split(r'.[\s\n]+\d+[.)] ', self.input_claims) 

        # remove numbering of the beginning of the input claims
        input_claims = [re.sub(r'^(\d+)[.)] ', '', claim).rstrip(" \n.") for claim in input_claims]

        # check if the generated claim is distinctive
        if generated_claim in input_claims:
            return False
        return True
    

    def check(self):

        # check if the numbering of the generated claim is coherent with the input claims
        numbering_coherence = self.numbering_coherence()
        # check if the dependency of the generated claim is correct as required
        depedency_correctness = self.depedency_correctness()
        # check if the punctuations of the generated claim is correct as required
        punctuations_correctness = self.punctuations_correctness()
        # check if is the case that only numbers are in parenthesis
        parenthesis_correctness = self.parenthesis_correctness()
        # check if the generated claim is hallucinated
        hallucination = self.no_hallucination()
        # check if the generated claim is distinctive
        distinctive = self.distinctive_claim()

        return {
            "numbering_coherence": numbering_coherence,
            "depedency_correctness": depedency_correctness,
            "punctuations_correctness": punctuations_correctness,
            "parenthesis_correctness": parenthesis_correctness,
            "hallucination": hallucination,
            "distinctive": distinctive
        }
    
    def score(self):
        # score the generated claim
        # return the score of the generated claim

        # check if the generated claim is valid
        results = self.check()

        score = 0
        if not results['distinctive']:  return score

        # score the generated claim
        for result in results.values():
            if result:
                score += 1
        return score / len(results)
        


        


