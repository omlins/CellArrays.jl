module Exceptions
export @IncoherentArgumentError, @ArgumentError
export IncoherentArgumentError, ArgumentError

macro IncoherentArgumentError(msg) esc(:(throw(IncoherentArgumentError($msg)))) end
macro ArgumentError(msg) esc(:(throw(ArgumentError($msg)))) end

struct IncoherentArgumentError <: Exception
    msg::String
end
Base.showerror(io::IO, e::IncoherentArgumentError) = print(io, "IncoherentArgumentError: ", e.msg)

struct ArgumentEvaluationError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ArgumentEvaluationError) = print(io, "ArgumentEvaluationError: ", e.msg)

end # Module Exceptions
