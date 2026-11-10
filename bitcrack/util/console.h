#ifndef _CONSOLE_H
#define _CONSOLE_H

#include <string>

// Códigos ANSI de cor para a interface do terminal.
// Funcionam em terminais compatíveis com ANSI (Linux, WSL2, macOS e
// Windows 10+ com VT habilitado). São std::string para facilitar a
// concatenação com as mensagens.
namespace Console {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string DIM     = "\033[2m";

    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string WHITE   = "\033[37m";
}

#endif
