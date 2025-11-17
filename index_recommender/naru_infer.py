class Naru:
    def __init__(self, ckpt, model_type="resmade"):
        self.model = load_model(ckpt, model_type)
        self.model.eval()

    def selectivity(self, predicate_list):
        prob = self.model.estimate_selectivity(predicate_list)
        return float(prob)

    def cardinality(self, predicate_list, table_rows):
        return self.selectivity(predicate_list) * table_rows
