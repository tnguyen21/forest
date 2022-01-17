import { React, useState } from "react";
import styled from "styled-components";

import { Box } from "@mui/material";

const SectionTitle = styled.h3`
  font-family: "Roboto", sans-serif;
  font-size: 1rem;
`;

const AnnotatedTextContainer = styled.div`
  font-family: "Roboto", sans-serif;
  line-height: 1.5;
`;

function helper() {
  /*
   * helper function that should probably exist to take text
   * and annotations and create some object to properly render
   * the annotated text ¯\_(ツ)_/¯
   */

  return 0;
}

function AnnotatedText({ outputText, annotations }) {
  // TODO some logic should be in place to style annotations

  return (
    <>
      <SectionTitle>Annotated Text</SectionTitle>
      <Box sx={{ width: "100%", height: 150 }}>
        <AnnotatedTextContainer>{outputText}</AnnotatedTextContainer>
      </Box>
    </>
  );
}

export default AnnotatedText;
