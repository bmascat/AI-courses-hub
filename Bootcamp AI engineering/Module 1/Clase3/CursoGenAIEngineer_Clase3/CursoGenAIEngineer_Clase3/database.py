import sqlite3

class BaseDatos:
    def __init__(self, nombre_db="interacciones.db"):
        self.conn = sqlite3.connect(nombre_db)
        self.cursor = self.conn.cursor()
        self.crear_tabla()

    def crear_tabla(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS interacciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                respuesta TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def guardar_interaccion(self, prompt, respuesta):
        self.cursor.execute("""
            INSERT INTO interacciones (prompt, respuesta)
            VALUES (?, ?)
        """, (prompt, respuesta))
        self.conn.commit()

    def obtener_interacciones(self, limite=5):
        self.cursor.execute("""
            SELECT prompt, respuesta FROM interacciones
            ORDER BY id DESC
            LIMIT ?
        """, (limite,))
        return self.cursor.fetchall()

    def cerrar_conexion(self):
        self.conn.close()

if __name__ == "__main__":
    db = BaseDatos()
    db.guardar_interaccion("Ejemplo de prompt", "Ejemplo de respuesta")
    interacciones = db.obtener_interacciones()
    for interaccion in interacciones:
        print(f"Prompt: {interaccion[0]}\nRespuesta: {interaccion[1]}\n")
    db.cerrar_conexion()
