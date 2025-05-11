import {
  AppBar,
  Toolbar,
  Button,
  Typography,
  Box,
  Menu,
  MenuItem,
  IconButton,
} from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { useState } from "react";
import AccountCircle from "@mui/icons-material/AccountCircle";

const Navigation = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleClose();
  };

  return (
    <AppBar
      position="static"
      sx={{
        width: "100%",
        backgroundColor: "#4c76cc", // Синій колір зі скріншота
        boxShadow: "none", // Без тіні для плоского дизайну
      }}
    >
      <Toolbar
        sx={{
          width: "100%",
          px: { xs: 2, sm: 3 },
          maxWidth: "100%",
        }}
      >
        <Typography
          variant="h6"
          component="div"
          sx={{
            flexGrow: 1,
            fontWeight: 400,
            fontSize: "1.1rem",
            letterSpacing: "0.5px",
          }}
        >
          Predictor
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center" }}>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
            disableRipple // Вимкнення ефекту ripple
            sx={{
              fontWeight: "normal",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              fontSize: "0.85rem",
              "&:hover": {
                backgroundColor: "transparent", // Вимкнення зміни фону при наведенні
                color: "white", // Залишаємо колір тексту білим
              },
            }}
          >
            ГОЛОВНА
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/forecast"
            disableRipple // Вимкнення ефекту ripple
            sx={{
              fontWeight: "normal",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              fontSize: "0.85rem",
              "&:hover": {
                backgroundColor: "transparent", // Вимкнення зміни фону при наведенні
                color: "white", // Залишаємо колір тексту білим
              },
            }}
          >
            ПРОГНОЗ
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/about"
            disableRipple // Вимкнення ефекту ripple
            sx={{
              fontWeight: "normal",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              fontSize: "0.85rem",
              "&:hover": {
                backgroundColor: "transparent", // Вимкнення зміни фону при наведенні
                color: "white", // Залишаємо колір тексту білим
              },
            }}
          >
            ПРО ДОДАТОК
          </Button>

          {isAuthenticated ? (
            <>
              <IconButton
                size="large"
                aria-label="account of current user"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleMenu}
                color="inherit"
              >
                <AccountCircle />
              </IconButton>
              <Menu
                id="menu-appbar"
                anchorEl={anchorEl}
                anchorOrigin={{
                  vertical: "bottom",
                  horizontal: "right",
                }}
                keepMounted
                transformOrigin={{
                  vertical: "top",
                  horizontal: "right",
                }}
                open={Boolean(anchorEl)}
                onClose={handleClose}
              >
                <MenuItem
                  component={RouterLink}
                  to="/profile"
                  onClick={handleClose}
                >
                  Профіль
                </MenuItem>
                <MenuItem onClick={handleLogout}>Вийти</MenuItem>
              </Menu>
            </>
          ) : (
            <>
              <Button
                color="inherit"
                component={RouterLink}
                to="/login"
                disableRipple
                sx={{
                  fontWeight: "normal",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  fontSize: "0.85rem",
                  "&:hover": {
                    backgroundColor: "transparent",
                    color: "white",
                  },
                }}
              >
                УВІЙТИ
              </Button>
              <Button
                color="inherit"
                component={RouterLink}
                to="/register"
                disableRipple
                sx={{
                  fontWeight: "normal",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  fontSize: "0.85rem",
                  "&:hover": {
                    backgroundColor: "transparent",
                    color: "white",
                  },
                }}
              >
                РЕЄСТРАЦІЯ
              </Button>
            </>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navigation;
