import transformers
from transformers import AutoModelForSequenceClassification
from transformers.pipelines.base import Dataset


def process_tree_output(tree_output):
    """
    Procesa la salida del comando `tree`.

    Args:
      tree_output: La salida del comando `tree`.

    Returns:
      Un diccionario que contiene el contexto del código del proyecto.
    """

    context = {}
    for line in tree_output.splitlines():
      if line.startswith("├──") or line.startswith("└──"):
        path = line.split(" ")[1]
        context[path] = {}
      elif line.startswith("|--"):
        path = line.split(" ")[1]
        context[path] = None
      elif line.startswith("-"):
        path = line.split(" ")[1]
        content = line.split(" ")[2:]
        context[path] = " ".join(content)
    return context


class CodeGPTInterface:

    def __init__(self, tree_output):
        # Crea la instancia del modelo de ChatGPT en el constructor
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained("microsoft/chatgpt3-medium")

        # Procesa la salida del comando `tree`
        self.context = process_tree_output(tree_output)

    def __call__(self):
        # Devuelve la instancia de la interfaz
        return self

    def generate_function(self, description):
        """
        Genera una función a partir de una descripción.

        Args:
          description: La descripción de la función que se desea generar.

        Returns:
          El código de la función generada.
        """

        tokens = description.split(" ")
        function_name = tokens[0]
        function_args = tokens[1:-1]
        function_body = tokens[-1]

        # Modifica el código para que use un conjunto de datos de código Java

        dataset = Dataset.load_dataset("text/code", "java")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/chatgpt3-medium")

        code = model.generate(
            text=description,
            do_sample=True,
            max_length=100,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.0,
            dataset=dataset,
        )

        return code
