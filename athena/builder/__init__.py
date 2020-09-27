from abc import ABC, abstractmethod
from typing import List, Union, Any, Callable


class Buildable(ABC):
    def __init__(self, parent: "Buildable", args: List[str] = None):
        """
        Class that provides methods that are often used in the builder pattern.
        Also provides some convenience functions that can be used to generate 
        the setter API methods automatically.

        Args:
            parent (Buildable): The parent builder interface. Can be None.
            args (List[str], optional): The list of names that are to be used to generate the setter methods\
                for example, if the list is ``["name"]``, a method with name ``name(value, *args, **kwargs)`` \ 
                will be generated, and when the method is called, the ``value`` is assigned to a variable called \
                ``_name``, which can be accessed via the getter ``get_name()``. The args passed can be accessed via \
                the getter ``get_name_args()`` and the keyword arguments passed can be accessed via the getter \
                ``get_name_kwargs()``. Defaults to None.
        """
        if args is not None and len(args) != 0:
            self._gen_builder_methods(args)
            self._gen_getters(args)
            self._gen_setters(args)

        self.parent = parent

        # context contains the variables that can be accessed
        # by child interfaces. For example, if a `model` was
        # defined by a parent interface, the child interface can
        # access it provided it was stored in the `context`.
        self.context = {}

    @abstractmethod
    def create(self) -> object:
        """
        Function that constructs and returns the required object.
        """

    def handle(self, obj: object):
        """
        Handles objects returned by the :func:`create` method of child interfaces.
        Can be stored in a local variable, etc.

        Args:
            obj (object): The object returned by the child interface.
        """
        pass

    def build(self) -> Union[object, "Buildable"]:
        """
        Builds the required object and returns it if ``parent`` is None.

        Returns:
            Union[object, Buildable]: The built object, if ``parent`` is None, \
                else the ``parent``.
        """

        # creating the object
        obj = self.create()

        # returning the object if `parent` is None
        if self.parent is None:
            return obj

        # else passing the object to the `parent` interface
        # so it can handle it
        self.parent.handle(obj)

        # and returning the parent.
        return self.parent

    def add_to_context(self, key: str, value: Any):
        """
        Stores a value in the interface's context.

        Args:
            key (str): The key used to access the value.
            value (Any): The value.
        """

        self.context[key] = value

    def find_in_context(self, key: str) -> Union[Any, None]:
        """
        Finds the value of the key in the current interface's context or
        in the parent's context.

        Args:
            key (str): The key of the required value.

        Returns:
            Union[Any, None]: The value of the object.
        """

        # getting the value from this interface's context
        value = self.context.get(key, None)

        # if the value is None and there is a parent,
        # checking in the parents context.
        if value is None and self.parent is not None:
            return self.parent.find_in_context(key)

        # returning the value
        return value

    def _gen_builder_methods(self, args: List[str]):
        """
        Generates the methods to be exposed as an API.

        Args:
            args (List[str]): The list of the names which the methods should have.
        """

        for arg in args:
            setattr(self, arg, self._gen_builder_method(arg))

    def _gen_getters(self, args: List[str]):
        """
        Generates the getter methods for the local variables that are generated
        to store the values passed from the exposed builder methods.

        Args:
            args (List[str]): The list of names of the exposed methods.
        """

        for arg in args:
            setattr(
                self,
                self._gen_instance_var_getter_name(arg),
                self._gen_getter(self._gen_instance_var_name(arg)),
            )
            setattr(
                self,
                self._gen_instance_var_args_getter_name(arg),
                self._gen_getter(self._gen_instance_var_args_name(arg)),
            )
            setattr(
                self,
                self._gen_instance_var_kwargs_getter_name(arg),
                self._gen_getter(self._gen_instance_var_kwargs_name(arg)),
            )

    def _gen_setters(self, args: List[str]):
        """
        Generates the setter methods for the local variables that are generated
        to store the values passed from the exposed builder methods.

        Args:
            args (List[str]): The list of names of the exposed methods.
        """
        for arg in args:
            # converting the given name to the name of the local variable
            var_name = self._gen_instance_var_name(arg)

            # assigning the method to the object
            setattr(
                self,
                self._gen_instance_var_setter_name(arg),
                self._gen_setter(var_name),
            )

    def _gen_builder_method(self, name: str) -> Callable:
        """
        Constructs a function that has the given name and assigns the values
        passed into it to local instance variables. These functions are the
        exposed APIs of the builder.

        Args:
            name (str): The name of the function.

        Returns:
            Callable: The constructed function.
        """

        def _wrapper(value, *args, **kwargs):
            setattr(self, self._gen_instance_var_name(name), value)
            setattr(self, self._gen_instance_var_args_name(name), args)
            setattr(self, self._gen_instance_var_kwargs_name(name), kwargs)

            return self

        return _wrapper

    def _gen_getter(self, name: str) -> Callable:
        """
        Constructs the getter function for the generated local instance variables.

        Args:
            name (str): The name of the generated local instance variable that has \
                to be returned by the getter.
        
        Returns:
            Callable: The constructed function.
        """

        def _wrapper():
            if hasattr(self, name):
                return getattr(self, name)
            return None

        return _wrapper

    def _gen_setter(self, name: str) -> Callable:
        """
        Constructs the setter function for the generated local instance variables.

        Args:
            name (str): The name of the generated local instance variable that has \
                to be set by the setter.
        
        Returns:
            Callable: The constructed function.
        """

        def _wrapper(value):
            setattr(self, name, value)

        return _wrapper

    def _gen_instance_var_name(self, name: str) -> str:
        """
        The name of the generated local instance variable.

        Args:
            name (str): The name of the function that has to be generated. \
                This function will store the value it recieves as a parameter \
                into a variable with the name this function returns.

        Returns:
            str: The name of the generated local instance variable.
        """

        return f"_{name}"

    def _gen_instance_var_args_name(self, name: str) -> str:
        """
        The name of the generated local instance variable to store the positional
        arguments that are passed into the exposed builder function.

        Args:
            name (str): The name of the function that has to be generated. \
                This function will store the additional positional arguments \
                it recieves as a parameter into a variable with the name this \
                function returns.

        Returns:
            str: The name of the generated local instance variable.
        """
        return f"_{name}_args"

    def _gen_instance_var_kwargs_name(self, name: str) -> str:
        """
        The name of the generated local instance variable to store the keyword
        arguments that are passed into the exposed builder function.

        Args:
            name (str): The name of the function that has to be generated. \
                This function will store the additional keyword arguments \
                it recieves as a parameter into a variable with the name this \
                function returns.

        Returns:
            str: The name of the generated local instance variable.
        """
        return f"_{name}_kwargs"

    def _gen_instance_var_getter_name(self, name: str) -> str:
        """
        The name of the getter of the generated local instance variable that
        stores the value passed into the exposed builder function.

        Args:
            name (str): The name of the exposed builder function.

        Returns:
            str: The name of the generated getter.
        """
        return f"get_{name}"

    def _gen_instance_var_args_getter_name(self, name: str) -> str:
        """
        The name of the getter of the generated local instance variable that
        stores the positional arguments passed into the exposed builder function.

        Args:
            name (str): The name of the exposed builder function.

        Returns:
            str: The name of the generated getter.
        """
        return f"get_{name}_args"

    def _gen_instance_var_kwargs_getter_name(self, name: str) -> str:
        """
        The name of the getter of the generated local instance variable that
        stores the keyword arguments passed into the exposed builder function.

        Args:
            name (str): The name of the exposed builder function

        Returns:
            str: The name of the generated getter.
        """
        return f"get_{name}_kwargs"

    def _gen_instance_var_setter_name(self, name: str) -> str:
        """
        The name of the setter of the generated local instance variable that
        sets the value passed into the exposed builder function.

        Args:
            name (str): The name of the exposed builder function

        Returns:
            str: The name of the generated setter.
        """
        return f"set_{name}"