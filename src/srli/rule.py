class Rule(object):
    def __init__(self, text, weight = None, options = {}, **kwargs):
        self._text = text
        self._weight = weight

        self._options = dict(options)
        self._options.update(kwargs)

    def text(self):
        return self._text

    def weight(self):
        return self._weight

    def is_weighted(self):
        return (self._weight is not None)

    def set_weight(self, weight):
        self._weight = weight

    def options(self):
        return self._options

    def __repr__(self):
        weight = '.'
        if (self.is_weighted()):
            weight = str(self._weight)

        return "%s: %s" % (weight, self._text)

    def to_dict(self):
        return {
            'text': self._text,
            'weight': self._weight,
            'options': self._options,
        }
