#include <iostream>
#include "TextLoader.h"

using namespace utec::data;

// global
TextLoader loader("training_words_esp.csv");

void load_data() {
    // Cargar y procesar los datos
    loader.load_data();

    // Obtener el dataset cargado
    const auto& dataset = loader.get_dataset();

    // Mostrar cuántos ejemplos se cargaron
    std::cout << "Cantidad de ejemplos cargados: " << dataset.size() << std::endl;

    // Mostrar tamaño del vocabulario
    std::cout << "Tamanho del vocabulario: " << loader.get_vocabulary_size() << std::endl;
}

// get mapping words of first sample
void show_map_of_first_sample() {
    const auto& dataset = loader.get_dataset();
    std::cout << "Map del primer dato del dataset: ";
    for (size_t i = 0; i < dataset[0].vectorized_text.size(); ++i) {
        if (dataset[0].vectorized_text[i]) std::cout << loader.get_vocabulary_list()[i] << " ";
    } std::cout << "\n";
}


int main() {
    load_data();
    show_map_of_first_sample();
    return 0;
}
