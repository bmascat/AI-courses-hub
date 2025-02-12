import requests

class ModeloGenerativo:
    def __init__(self, nombre, version):
        self._nombre = nombre
        self._version = version

    @property
    def nombre(self):
        return self._nombre

    @nombre.setter
    def nombre(self, valor):
        self._nombre = valor

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, valor):
        self._version = valor

    def cargar_modelo(self):
        # Código para cargar el modelo
        pass

    def generar_texto(self, prompt, variables=None):
        """
        Método genérico para generar texto basado en el prompt.
        Si se proporciona un diccionario de variables, se formatea el prompt.
        """
        if variables:
            prompt = self.formatear_prompt(prompt, variables)
        prompt_limpio = self.limpiar_prompt(prompt)
        # Aquí se debería implementar la lógica de generación; en esta versión se simula la respuesta.
        texto_generado = f"Texto generado por {self.nombre} versión {self.version} para el prompt: {prompt_limpio}"
        return texto_generado

    def limpiar_prompt(self, prompt):
        prompt = prompt.strip()
        prompt = prompt.capitalize()
        return prompt

    def formatear_prompt(self, plantilla, variables):
        try:
            prompt_formateado = plantilla.format(**variables)
            return prompt_formateado
        except KeyError as e:
            print(f"Error: Falta la variable {e} en el diccionario de variables.")
            return plantilla

    def crear_one_shot_prompt(self, tarea, ejemplo):
        return f"Ejemplo:\nTarea: {ejemplo['tarea']}\nRespuesta: {ejemplo['respuesta']}\n\nTarea: {tarea}\nRespuesta:"

    def crear_many_shots_prompt(self, tarea, ejemplos):
        prompt = ""
        for i, ejemplo in enumerate(ejemplos, 1):
            prompt += f"Ejemplo {i}:\nTarea: {ejemplo['tarea']}\nRespuesta: {ejemplo['respuesta']}\n\n"
        prompt += f"Tarea: {tarea}\nRespuesta:"
        return prompt

class ModeloGPT(ModeloGenerativo):
    def __init__(self, nombre, version, api_key, db):
        super().__init__(nombre, version)
        self.api_key = api_key
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.db = db

    def generar_texto(self, prompt, variables=None):
        if variables:
            prompt = self.formatear_prompt(prompt, variables)
        prompt_limpio = self.limpiar_prompt(prompt)
        # Create the messages list as required by the chat completions endpoint
        messages = [
            {"role": "system", "content": "Eres un asistente de IA que ayuda a generar textos."},
            {"role": "user", "content": prompt_limpio}
        ]
        payload = {
            "model": "gpt-3.5-turbo",  # Use a model that is supported
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        if response.status_code == 200:
            texto_generado = response.json()["choices"][0]["message"]["content"].strip()
            self.db.guardar_interaccion(prompt_limpio, texto_generado)
            return texto_generado
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def generar_texto_con_cot(self, prompt, cot_steps):
        prompt_cot = f"""
        {prompt}

        Chain of Thought:
        {cot_steps}

        Respuesta:
        """
        prompt_limpio = self.limpiar_prompt(prompt_cot)
        payload = {
            "model": "text-davinci-003",
            "prompt": prompt_limpio,
            "max_tokens": 150,
            "temperature": 0.7
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        if response.status_code == 200:
            texto_generado = response.json()["choices"][0]["text"].strip()
            self.db.guardar_interaccion(prompt_limpio, texto_generado)
            return texto_generado
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    def generar_texto_con_cot_many_shots(self, tarea, ejemplos, cot_steps_list):
        prompt = "Contexto: Eres un asistente de IA que ayuda a resolver problemas complejos paso a paso.\n\n"
        for i, (ejemplo, cot_steps) in enumerate(zip(ejemplos, cot_steps_list), 1):
            prompt += f"Ejemplo {i}:\n"
            prompt += f"Tarea: {ejemplo['tarea']}\n"
            prompt += f"Chain of Thought:\n{cot_steps}\nRespuesta: {ejemplo['respuesta']}\n\n"
        prompt += f"Tarea: {tarea}\nChain of Thought:\n"
        return self.generar_texto(prompt)
