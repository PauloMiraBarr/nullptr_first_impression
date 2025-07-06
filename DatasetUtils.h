//
// Created by paulo on 5/07/2025.
//

#ifndef DATASETUTILS_H
#define DATASETUTILS_H

#include "tensor.h"
#include "TextLoader.h"

namespace utec::data {

    class DatasetUtils {
    public:
        static utec::algebra::Tensor<float, 2> vector_to_tensor(const std::vector<TextExample>& dataset);
        static utec::algebra::Tensor<float, 2> labels_to_tensor(const std::vector<TextExample>& dataset);

        // split dataset: dividir los datos entre entrenamiento y prueba
        static void split_dataset(const std::vector<TextExample>& dataset,
                                  std::vector<TextExample>& train_set,
                                  std::vector<TextExample>& test_set,
                                  float train_ratio = 0.8);
    };

}

#endif //DATASETUTILS_H
