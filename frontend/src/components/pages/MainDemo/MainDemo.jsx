import { React, useState } from "react";
import styled from "styled-components";

import { exampleText } from "./constants";

import Container from "@mui/material/Container";
import { Box, TextField, Typography } from "@mui/material";

const PageTitle = styled.h2`
  font-family: "Roboto", sans-serif;
  font-size: 1.5rem;got
`;

const inputStyles = {
  style: {
    fontSize: ".8rem",
    fontFamily: "Roboto mono",
  },
};

function MainDemo() {
  const [inputText, setInputText] = useState(exampleText);

  return (
    <Container>
      <PageTitle>Main Demo</PageTitle>
      <Box sx={{ width: "100%", height: 150 }}>
        {" "}
        <Typography>Explanation of demo goes here.</Typography>
      </Box>
      <Box
        sx={{
          "& .MuiTextField-root": { m: 1, maxWidth: "100%" },
        }}
      >
        <TextField
          fullWidth
          label="Text Input"
          multiline
          rows={6}
          inputProps={inputStyles}
          defaultValue={inputText}
          variant="standard"
        />
      </Box>
      <Box sx={{ mt: "3rem", width: "100%", height: 150 }}>
        <Typography>Output goes here.</Typography>
      </Box>
    </Container>
  );
}

export default MainDemo;
