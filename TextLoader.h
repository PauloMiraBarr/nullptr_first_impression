//
// Created by paulo on 3/07/2025.
//

#ifndef TEXTLOADER_H
#define TEXTLOADER_H

#include <string>
#include <vector>
#include <unordered_map>

namespace utec::data {

    // Definimos de lo que se estructurara una fila (o dato)
    struct TextExample {
        std::vector<float> vectorized_text;
        int label;
    };

    // Clase encargada de cargar el dataset
    class TextLoader {
    private:
        // Atributos
        std::string filename_;
        std::vector<TextExample> dataset_;
        std::unordered_map<std::string, int> vocabulary_;
        std::vector<std::string> vocabulary_list_;

        // Métodos internos
        void build_vocabulary();
        std::vector<std::string> tokenize(const std::string& text);


    public:
        TextLoader() = default;
        TextLoader(const std::string& filename);
        void load_data();
        const std::vector<TextExample>& get_dataset() const;
        size_t get_vocabulary_size() const;
        int get_label(const std::string& label_text);
        std::vector<float> vectorize(const std::string& text);
        const std::vector<std::string>& get_vocabulary_list() const;
    };

}




#endif //TEXTLOADER_H
