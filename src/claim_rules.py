# rules for checking the validity of a generated claim given previous claims
import re


class Rule_based_checker(object):
    def __init__(self, input_claims, generated_claim, required_dependent=True):
        self.input_claims = input_claims
        self.generated_claim = generated_claim
        self.required_dependent = required_dependent
        self.trans_phrases = ["further_comprising", "furthre_comprises", "further_configured_to", "comprising", "consisting_of", "wherein", "in_which", "whereby", "such_that", "so_as_to", "characterized_in_that"]

        # extract the numberings of the input claims
        self.input_numberings = re.findall(r'[\s\n]?(\d+)[.)] ', self.input_claims)
        self.input_numberings = [int(numbering) for numbering in self.input_numberings if numbering != '']


    def numbering_coherence(self):
        """
        check if the numbering of the generated claim is coherent with the input claims
        return True if the numbering is coherent, False otherwise
        e.g. input claims: [1, 2, 3, 4, 5], generated claim: 6, return True
        e.g. input claims: [1, 2, 3, 4, 5], generated claim: 7, return False
        """

        # extract the numbering of the generated claim
        generated_numbering = re.findall(r'[\s\n]?(\d+)[.)] ', self.generated_claim)
        if len(generated_numbering) == 0:
            return False

        generated_numbering = int(generated_numbering[0]) if generated_numbering[0] != '' else int(generated_numbering[1])

        # check if the input claims are not empty
        assert len(self.input_claims) != 0 and len(self.input_numberings) != 0

        if generated_numbering != self.input_numberings[-1] + 1:
            return False
        return True
    

    def dependency_correctness(self):
        """ 
        check if the dependency of the generated claim is correct as required
        return True if the dependency is correct, False otherwise
        """

        # extract the dependency of the generated claim
        is_dependent = (not re.search("^\d+. A |^\d+. An ", self.generated_claim)) or ("according to" in self.generated_claim and re.search(r"\sclaims?[\s,]", self.generated_claim))

        if self.required_dependent and is_dependent is None:
            return False
        if (not self.required_dependent) and (is_dependent is not None):
            return False
        
        if is_dependent:
            # extract the numbering of the dependent claim
            rereferences = re.compile(r'(Claims? (?P<range_start>\d+)(?: to (?:Claims? )?|-)(?P<range_end>\d+))|(Claims? (?P<or_claim1>\d+)((, (?P<or_claim_list>\d+))* or (?:Claims? )?(?P<or_claim2>\d+)))|Claim (?P<single_claim>\d+)|(?P<any_preceding>any (?:(?:one )?of the )?preceding Claims)|(?P<aforementioned>one of the aforementioned claims)', re.IGNORECASE)
            dependent_claim_numbering = []
            for m in rereferences.finditer(self.generated_claim):
                if m.group('range_start') and m.group('range_end'):
                    dependent_claim_numbering = list(range(int(m.group('range_start')), int(m.group('range_end'))+1))
                elif m.group('or_claim1') and m.group('or_claim2'):
                    dependent_claim_numbering = [int(m.group('or_claim1')), int(m.group('or_claim2'))]
                elif m.group('single_claim'):
                    dependent_claim_numbering = [int(m.group('single_claim'))]
                elif m.group('any_preceding') or m.group('aforementioned'):
                    dependent_claim_numbering = list(range(1, self.input_numberings[-1]))

            if len(dependent_claim_numbering) == 0:
                dependent_claim_numbering = re.findall(r'(\d+)', self.generated_claim.split('claims')[-1])
                if len(dependent_claim_numbering) == 0:
                    return False

            # check if the dependency of the generated claim is in the input claims
            dependent_claim_numbering = [int(numbering) for numbering in dependent_claim_numbering]
            for n in dependent_claim_numbering:
                if n not in self.input_numberings:
                    return False

        return True
    
    
    def _befaft(self, s, words):  # splits at first occurrence of word from words
        reSplit = re.compile(
            r'(\b'+r'\b|\b'.join([w.replace('_', ' ') for w in words.split()])+r'\b)', re.I)
        return [t.strip() for t in (reSplit.split(s, maxsplit=1)+['', ''])[:3]]


    def punctuations_correctness(self):
        """
        check if the punctuations of the generated claim is correct as required
        return True if the punctuations is correct, False otherwise
        """

        # extract the punctuations of the generated claim
        punctuations = re.findall(r'[.,;:!?]', self.generated_claim)
        if len(punctuations) == 0:
            return False
        if self.generated_claim.strip()[-1] != '.':
            return False

        # split by list in trans_phrases
        before, _, _ = self._befaft(self.generated_claim, " ".join(self.trans_phrases))

        if len(before) > 0 and before.strip()[-1] != ',':
            return False
        return True


    def _remove_repetitive_spans(self):
        # Define a regular expression pattern to find repetitive spans of at least 2 repetitions.
        pattern = r'(?<![A-Za-z0-9])(.{2,}?)\1{1,}(?![A-Za-z0-9])'

        # Use re.sub to remove all matching repetitive spans in the text, except for the first instance.
        first_instance = re.sub(pattern, r'\1', self.generated_claim)

        return first_instance


    def no_hallucination(self):
        """
        check if the generated claim is hallucinated (repetition of phrases over three times)
        return True if the generated claim is not hallucinated, False otherwise
        """

        # remove repetitive spans
        cleaned_generation = self._remove_repetitive_spans()
        # check if the generated claim is hallucinated
        if cleaned_generation == self.generated_claim:
            return True
        return False
    

    def distinctive_claim(self):
        """
        check if the generated claim is distinctive
        return True if the generated claim is distinctive, False otherwise
        """

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
        dependency_correctness = self.dependency_correctness()
        # check if the punctuations of the generated claim is correct as required
        punctuations_correctness = self.punctuations_correctness()
        # check if the generated claim is hallucinated
        no_hallucination = self.no_hallucination()
        # check if the generated claim is distinctive
        distinctive = self.distinctive_claim()

        return {
            "numbering_coherence": numbering_coherence,
            "dependency_correctness": dependency_correctness,
            "punctuations_correctness": punctuations_correctness,
            "no_hallucination": no_hallucination,
            "distinctive": distinctive
        }
    
    def score(self):
        # score the generated claim
        # return the score of the generated claim

        # check if the generated claim is valid
        results = self.check()

        score = 0
        if not results['distinctive']:  
            return score    # if the generated claim is not distinctive, return 0
        else:
            # score the generated claim
            for result in results.values():
                if result:
                    score += 1
            return score / len(results)
            


        


