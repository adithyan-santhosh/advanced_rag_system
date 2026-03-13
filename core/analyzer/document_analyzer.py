import re


class DocumentAnalyzer:

    def __init__(self, text):
        self.text = text

    def is_code_heavy(self):
        tokens = self.text.split()
        total_tokens = len(tokens)

        if total_tokens == 0:
            return False

        digit_tokens = 0
        pattern_tokens = 0
        long_alnum_tokens = 0

        for token in tokens:

            # Contains digit
            if any(char.isdigit() for char in token):
                digit_tokens += 1

            # Pattern like INV-2024-AX9347
            if re.search(r"[A-Z]{2,}-\d+", token):
                pattern_tokens += 1

            # Long alphanumeric token
            if re.match(r"[A-Z0-9]{6,}", token):
                long_alnum_tokens += 1

        digit_ratio = digit_tokens / total_tokens

        # Heuristic thresholds
        if digit_ratio > 0.15:
            return True

        if pattern_tokens > 3:
            return True

        if long_alnum_tokens > 5:
            return True

        return False
