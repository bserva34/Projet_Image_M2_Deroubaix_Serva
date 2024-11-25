from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


class MyApp(App):
    def build(self):
        # Crée un layout principal (vertical)
        layout = BoxLayout(orientation='vertical')

        # Ajoute un bouton au layout
        button = Button(text="Cliquez-moi", size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5})
        button.bind(on_press=self.on_button_click)  # Associe un événement au bouton
        layout.add_widget(button)

        return layout

    def on_button_click(self, instance):
        print("Bouton cliqué !")


if __name__ == '__main__':
    MyApp().run()
