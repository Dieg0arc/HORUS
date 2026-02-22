class UserManager:
    """Gestiona la información del usuario actual de forma simplificada."""
    
    def __init__(self):
        self.current_user = "Invitado"

    def set_user(self, name):
        """Establece el nombre del niño que está jugando."""
        self.current_user = name.strip() if name.strip() else "Invitado"

    def get_user_name(self):
        """Obtiene el nombre del niño del usuario actual."""
        return self.current_user

    def update_stats(self, score):
        """Método simplificado (no hace persistencia en JSON si no se requiere)."""
        pass

# Instancia única para ser compartida entre escenas
user_manager = UserManager()
