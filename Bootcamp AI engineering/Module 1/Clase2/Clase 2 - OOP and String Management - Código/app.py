from modelo_generativo import ModeloGPT
from database import BaseDatos

def main():
    api_key = "API_KEY"
    db = BaseDatos()
    modelo = ModeloGPT("GPT-3", "v1.0", api_key, db)

    print("=== Generador de Textos Personalizado con GenAI ===")
    while True:
        plantilla = input("Ingresa una plantilla de prompt (o 'salir' para terminar): ")
        if plantilla.lower() == 'salir':
            break
        try:
            num_variables = int(input("¿Cuántas variables deseas ingresar? "))
        except ValueError:
            print("Por favor, ingresa un número válido.")
            continue
        variables = {}
        for _ in range(num_variables):
            clave = input("Nombre de la variable: ")
            valor = input(f"Valor para '{clave}': ")
            variables[clave] = valor

        texto_generado = modelo.generar_texto(plantilla, variables)
        if texto_generado:
            print(f"\n**Respuesta Generada:**\n{texto_generado}\n{'-'*50}")
        else:
            print("No se pudo generar el texto.")

    print("Guardando y cerrando la aplicación...")
    db.cerrar_conexion()

if __name__ == "__main__":
    main()
