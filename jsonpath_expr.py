"""JSONPathExpr class for working with JSONPath expressions and their polars equivalents."""

from typing import Optional

# Import the conversion function from the main module
from jsonpath_to_polars import jsonpath_to_polars


class JSONPathExpr:
    """A class to work with JSONPath expressions and their polars equivalents.
    
    This class provides an interface between JSONPath string expressions and polars
    Expr objects, with methods to view and manipulate both representations.
    
    Args:
        jsonpath: A string containing a valid JSONPath expression
        alias: An optional name for the expression in the resulting schema
    
    Attributes:
        jsonpath: The original JSONPath string
        expr: The corresponding polars Expr
        alias: The name for this expression in the resulting schema
    """
    
    def __init__(self, jsonpath: str, alias: Optional[str] = None):
        """Initialize with a JSONPath expression and optional alias.
        
        Args:
            jsonpath: A string containing a valid JSONPath expression
            alias: An optional name for the expression in the resulting schema
        """
        self.jsonpath = jsonpath
        self.alias = alias
        
        # Convert the JSONPath to a polars expression
        self.expr = jsonpath_to_polars(jsonpath)
        
        # Apply alias if provided
        if alias is not None:
            self.expr = self.expr.alias(alias)
    
    def __str__(self) -> str:
        """Return a string representation of the JSONPathExpr.
        
        Returns:
            A string showing both the JSONPath and its polars representation
        """
        return f"JSONPath: {self.jsonpath}\nPolars: {self.expr_str()}"
    
    def __repr__(self) -> str:
        """Return a representation of the JSONPathExpr.
        
        Returns:
            A string representation including JSONPath and alias
        """
        alias_str = f", alias='{self.alias}'" if self.alias else ""
        return f"JSONPathExpr('{self.jsonpath}'{alias_str})"
    
    def jsonpath_str(self) -> str:
        """Return the original JSONPath string.
        
        Returns:
            The JSONPath string
        """
        return self.jsonpath
    
    def expr_str(self) -> str:
        """Return the string representation of the polars Expr.
        
        Returns:
            The string representation of the polars expression
        """
        return str(self.expr)
    
    def tree_diagram(self) -> str:
        """Return the compute tree diagram of the polars Expr.
        
        Returns:
            The string representation of the expression tree diagram
        """
        return self.expr.meta.tree_format(return_as_string=True)
    
    def with_alias(self, alias: str) -> 'JSONPathExpr':
        """Return a new JSONPathExpr with the given alias.
        
        Args:
            alias: The new alias for the expression
            
        Returns:
            A new JSONPathExpr instance with the updated alias
        """
        return JSONPathExpr(self.jsonpath, alias)