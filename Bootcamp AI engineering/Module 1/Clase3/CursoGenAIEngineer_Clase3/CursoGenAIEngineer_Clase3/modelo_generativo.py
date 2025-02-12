import requests
from typing import Dict, List
import json
import time

class ModeloGenerativo:
    def __init__(self, nombre, version):
        self._nombre = nombre
        self._version = version
        self._estilos_escritura = {
            "poético": "Escribe en un estilo poético y lírico",
            "técnico": "Usa un tono técnico y preciso",
            "narrativo": "Cuenta una historia envolvente",
            "periodístico": "Escribe como un artículo periodístico",
            "humorístico": "Usa un tono divertido y ligero"
        }

    @property
    def nombre(self):
        return self._nombre

    @property
    def estilos_disponibles(self):
        return list(self._estilos_escritura.keys())

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
        if variables:
            prompt = self.formatear_prompt(prompt, variables)
        return f"Texto generado para: {prompt}"

    def formatear_prompt(self, plantilla, variables):
        try:
            return plantilla.format(**variables)
        except KeyError as e:
            print(f"Error: Falta la variable {e}")
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
        self.db = db
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generar_texto(self, prompt, variables=None):
        if variables:
            prompt = self.formatear_prompt(prompt, variables)

        messages = [
            {"role": "system", "content": "Eres un asistente de IA que ayuda a generar textos."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "max_tokens": 150,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                texto_generado = response.json()["choices"][0]["message"]["content"].strip()
                self.db.guardar_interaccion(prompt, texto_generado)
                return texto_generado
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def generar_imagen(self, prompt):
        """Genera una imagen usando DALL-E"""
        try:
            response = requests.post(
                f"{self.base_url}/images/generations",
                headers=self.headers,
                json={
                    "prompt": prompt,
                    "n": 1,
                    "size": "1024x1024"
                }
            )
            
            if response.status_code == 200:
                url_imagen = response.json()["data"][0]["url"]
                self.db.guardar_interaccion(f"[IMAGEN] {prompt}", url_imagen)
                return url_imagen
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error: {str(e)}")
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
