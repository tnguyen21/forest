import { React, useState } from "react";
import styled from "styled-components";

import { Box, Divider } from "@mui/material";
import Annotation from "./Annotation";

const AnnotatedTextContainer = styled.div`
  padding: 1.2rem 1rem;
  border: 1px solid #e0e0e0;
  font-family: "Roboto mono", sans-serif;
  line-height: 1.5;
`;

const AnnotationDescription = styled.div`
  font-family: "Roboto mono", sans-serif;
  padding: 0.8rem 1rem;
`;

function AnnotatedText({ outputText, annotations }) {
  console.log("annotations", annotations);
  const getHighlightedText = (text, highlight) => {
    // Split text on highlight term, include term itself into parts, ignore case
    const parts = text.split(new RegExp(`(${highlight})`, "gi"));
    return (
      <>
        <AnnotatedTextContainer variant="outlined">
          {parts.map((part, idx) =>
            part.toLowerCase() === highlight.toLowerCase() ? (
              <Annotation key={`annotation-${idx}`} text={part} />
            ) : (
              part
            )
          )}
        </AnnotatedTextContainer>
        <Divider />
        {annotations &&
          Object.entries(annotations).map((annotation, idx) => {
            const [text, category] = annotation;
            console.log(idx);
            return (
              <AnnotationDescription key={`desc-${idx}`}>
                {text} - {category}
              </AnnotationDescription>
            );
          })}
      </>
    );
  };

  return (
    <Box sx={{ width: "100%", mb: 5 }}>
      {getHighlightedText(outputText, "Freud")}
    </Box>
  );
}

export default AnnotatedText;
