#include <iostream>
#include "TextLoader.h"

using namespace utec::data;

int main() {
    // Paso 1: Crear el loader y pasarle el nombre del archivo CSV
    TextLoader loader("training_words_esp.csv");

    // Paso 2: Cargar y procesar los datos
    loader.load_data();

    // Paso 3: Obtener el dataset cargado
    const auto& dataset = loader.get_dataset();

    // Paso 4: Mostrar cuántos ejemplos se cargaron
    std::cout << "Cantidad de ejemplos cargados: " << dataset.size() << std::endl;

    // Paso 5: Mostrar tamaño del vocabulario
    std::cout << "Tamanho del vocabulario: " << loader.get_vocabulary_size() << std::endl;

    // Paso 6: Mostrar los primeros ejemplos (puedes imprimir 3 como prueba)
    for (size_t i = 0; i < 3 && i < dataset.size(); ++i) {
        std::cout << "[" << i + 1 << "] ";
        std::cout << "Etiqueta: " << dataset[i].label << ", Vector: ";

        // Mostrar primeros 10 valores del vector para no saturar la pantalla
        for (size_t j = 0; j < 10 && j < dataset[i].vectorized_text.size(); ++j) {
            std::cout << dataset[i].vectorized_text[j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}
