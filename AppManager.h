//
// Created by paulo on 5/07/2025.
//

#ifndef APPMANAGER_H
#define APPMANAGER_H

namespace utec::app {
    class AppManager {
    public:
        void show_menu();

    private:
        void train_model();
        void test_model();
        void predict_message();
        void run_tests();
    };
}



#endif //APPMANAGER_H
