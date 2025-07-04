//
// Created by paulo on 3/07/2025.
//

#include "TextLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <algorithm>

using namespace utec::data;

TextLoader::TextLoader(const std::string& filename) : filename_(filename) {}

void TextLoader::load_data() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo: " << filename_ << std::endl;
        return;
    }

    std::string line;

    // Primer recorrido: construir vocabulario
    build_vocabulary();

    // Segundo recorrido: vectorizar mensajes
    file.clear();
    file.seekg(0); // Volver al inicio del archivo

    // Ignorar cabecera
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label_text, message;

        std::getline(ss, label_text, ',');
        std::getline(ss, message);

        TextExample example;
        example.label = get_label(label_text);
        example.vectorized_text = vectorize(message);

        dataset_.push_back(example);
    }

    file.close();
}

void TextLoader::build_vocabulary() {
    std::ifstream file(filename_);
    std::string line;

    // Ignorar cabecera
    std::getline(file, line);

    std::unordered_set<std::string> unique_words;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string label_text, message;

        std::getline(ss, label_text, ',');
        std::getline(ss, message);

        auto words = tokenize(message);
        unique_words.insert(words.begin(), words.end());
    }

    // Mapea las palabras con un índice
    int index = 0;
    for (const auto& word : unique_words) {
        vocabulary_[word] = index++;
    }

    file.close();
}

std::vector<std::string> TextLoader::tokenize(const std::string& text) {
    std::stringstream ss(text);
    std::string word;
    std::vector<std::string> tokens;

    while (ss >> word) {
        // Normalizamos quitando signos y poniendo en minúsculas
        word.erase(std::remove_if(word.begin(), word.end(), ispunct), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        tokens.push_back(word);
    }

    return tokens;
}

std::vector<float> TextLoader::vectorize(const std::string& text) {
    std::vector<float> vector_frecuency(vocabulary_.size(), 0.0f);
    auto words = tokenize(text);

    for (const auto& word : words) {
        if (vocabulary_.find(word) != vocabulary_.end()) {
            vector_frecuency[vocabulary_[word]] += 1.0f; // Bag of words simple (cantidad) para hacerlo por presencia basta cambiarlo a = en vez de  +=
        }
    }

    return vector_frecuency;
}

int TextLoader::get_label(const std::string& label_text) {
    return (label_text == "spam");
}

const std::vector<TextExample>& TextLoader::get_dataset() const {
    return dataset_;
}

size_t TextLoader::get_vocabulary_size() const {
    return vocabulary_.size();
}

