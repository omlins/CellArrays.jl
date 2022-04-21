module Exceptions
export @ModuleInternalError, @IncoherentArgumentError, @ArgumentError
export ModuleInternalError, IncoherentArgumentError, ArgumentError

macro ModuleInternalError(msg) esc(:(throw(ModuleInternalError($msg)))) end
macro IncoherentArgumentError(msg) esc(:(throw(IncoherentArgumentError($msg)))) end
macro ArgumentError(msg) esc(:(throw(ArgumentError($msg)))) end

struct ModuleInternalError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ModuleInternalError) = print(io, "ModuleInternalError: ", e.msg)

struct IncoherentArgumentError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IncoherentArgumentError) = print(io, "IncoherentArgumentError: ", e.msg)

struct ArgumentEvaluationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ArgumentEvaluationError) = print(io, "ArgumentEvaluationError: ", e.msg)

end # Module Exceptions
