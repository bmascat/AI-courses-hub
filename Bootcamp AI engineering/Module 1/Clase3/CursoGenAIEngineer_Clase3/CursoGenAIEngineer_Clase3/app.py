from modelo_generativo import ModeloGPT
from database import BaseDatos

def mostrar_menu():
    print("\n=== MENU ===")
    print("1. Generar texto")
    print("2. Generar imagen")
    print("3. Generar texto e imagen")
    print("4. Salir")
    return input("Selecciona una opción: ")

def obtener_variables(num_variables):
    variables = {}
    for _ in range(num_variables):
        clave = input("Nombre de la variable: ")
        valor = input(f"Valor para '{clave}': ")
        variables[clave] = valor
    return variables

def main():
    api_key = ""
    db = BaseDatos()
    modelo = ModeloGPT("GPT-3", "v1.0", api_key, db)

    print("=== Generador de Contenido con GenAI ===")
    
    while True:
        opcion = mostrar_menu()
        
        if opcion == "4":
            break
            
        if opcion in ["1", "2", "3"]:
            if opcion in ["1", "3"]:
                plantilla = input("Ingresa una plantilla de prompt: ")
                try:
                    num_variables = int(input("¿Cuántas variables deseas ingresar? "))
                    variables = obtener_variables(num_variables)
                except ValueError:
                    print("Por favor, ingresa un número válido.")
                    continue
                
                texto_generado = modelo.generar_texto(plantilla, variables)
                if texto_generado:
                    print(f"\n**Texto Generado:**\n{texto_generado}")
                else:
                    print("No se pudo generar el texto.")

            if opcion in ["2", "3"]:
                if opcion == "2":
                    prompt_imagen = input("Describe la imagen que deseas generar: ")
                else:
                    prompt_imagen = input("¿Deseas usar un prompt específico para la imagen? (Enter para usar el texto generado): ")
                    if not prompt_imagen:
                        prompt_imagen = texto_generado

                print("\nGenerando imagen...")
                url_imagen = modelo.generar_imagen(prompt_imagen)
                
                if url_imagen:
                    print(f"\n**Imagen generada:**\n{url_imagen}")
                    print("Puedes abrir este enlace en tu navegador para ver la imagen.")
                else:
                    print("No se pudo generar la imagen.")

            print("-"*50)
        else:
            print("Opción no válida")

    print("Guardando y cerrando la aplicación...")
    db.cerrar_conexion()

if __name__ == "__main__":
    main()